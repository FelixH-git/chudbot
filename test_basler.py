import cv2
import numpy as np
from pypylon import pylon

aruco_length = 100  # mm

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


def detect_shape(cnt):
    """Classify a contour as square, rectangle, triangle, hexagon, star, circle, etc."""
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return "Unknown"

    # Approximate the contour to get its main vertices
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    vertices = len(approx)

    # Circularity: 1.0 = perfect circle
    area = cv2.contourArea(cnt)
    circularity = 4 * np.pi * area / (peri * peri)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # A square has roughly equal width and height
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h != 0 else 0
        if 0.9 <= aspect <= 1.1:
            return "Square"
        return "Rectangle"
    elif vertices == 6:
        return "Hexagon"
    elif 8 <= vertices <= 12 and circularity < 0.8:
        # A 5-pointed star approximates to ~10 sharp corners
        return "Star"
    elif circularity > 0.85:
        return "Circle"
    return f"Polygon({vertices})"

# Converter to turn raw Bayer sensor data into a color BGR array
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

with pylon.InstantCamera(pylon.FirstFound) as camera:
    print("Using device:", camera.DeviceInfo.ModelName)

    camera.Width.TrySetToMaximum()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    while camera.IsGrabbing():
        try:
            with camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            ) as grab_result:
                if grab_result.GrabSucceeded():
                    # Convert raw frame to color BGR numpy array
                    image = converter.Convert(grab_result)
                    img = image.GetArray()

                    # 1. Detect ArUco markers on the original full-size image
                    corners, ids, _ = aruco_detector.detectMarkers(img)

                    # 2. Resize the image down
                    img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

                    # 3. Apply the color mask
                    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                    lower_color = np.array([0, 50, 50])
                    upper_color = np.array([179, 255, 255])
                    mask = cv2.inRange(hsv, lower_color, upper_color)

                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                    filtered_img = cv2.bitwise_and(img_resized, img_resized, mask=mask)

                    # 4. Draw markers, text, and object distances after the mask is applied
                    if len(corners) > 0:
                        # Scale full-size corners down by 0.5 to match the resized image
                        scaled_corners = [c * 0.5 for c in corners]
                        cv2.aruco.drawDetectedMarkers(filtered_img, scaled_corners, ids)
                        
                        aruco_perimeter = cv2.arcLength(scaled_corners[0], True)
                        pixel_cm_ratio = aruco_perimeter / (aruco_length * 4)

                        # Use the top-left corner of the first ArUco marker as the reference point
                        # (corners are ordered [top-left, top-right, bottom-right, bottom-left])
                        aruco_ref = scaled_corners[0][0][0]
                        aruco_ref_int = (int(aruco_ref[0]), int(aruco_ref[1]))
                        
                        # Draw the marker reference (corner) point
                        cv2.circle(filtered_img, aruco_ref_int, 5, (255, 0, 0), -1)

                        # Find contours of colored objects separated by black background
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        object_id = 0  # Running ID for each detected color region
                        for cnt in contours:
                            # Filter out small noise areas
                            if cv2.contourArea(cnt) > 300:
                                object_id += 1
                                # Find the center (centroid) of the object using image moments
                                M = cv2.moments(cnt)
                                if M["m00"] != 0:
                                    obj_cx = int(M["m10"] / M["m00"])
                                    obj_cy = int(M["m01"] / M["m00"])

                                    # Classify the shape of the colored object
                                    shape = detect_shape(cnt)
                                    
                                    # Calculate distance in pixels from the ArUco reference corner
                                    dx_px = obj_cx - aruco_ref[0]
                                    dy_px = obj_cy - aruco_ref[1]
                                    
                                    # Convert pixel delta to millimeters
                                    dx_mm = dx_px / pixel_cm_ratio
                                    dy_mm = dy_px / pixel_cm_ratio
                                    total_dist_mm = np.sqrt(dx_mm**2 + dy_mm**2)

                                    # Draw line from the ArUco corner to the object center
                                    cv2.line(filtered_img, aruco_ref_int, (obj_cx, obj_cy), (255, 255, 0), 2)
                                    # Draw object center point
                                    cv2.circle(filtered_img, (obj_cx, obj_cy), 5, (0, 0, 255), -1)
                                    # Draw the contour outline
                                    cv2.drawContours(filtered_img, [cnt], -1, (0, 255, 0), 2)

                                    # Display the object ID above the center point
                                    cv2.putText(
                                        filtered_img,
                                        f"ID: {object_id}",
                                        (obj_cx - 10, obj_cy - 55),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 255, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )

                                    # Display the detected shape below the center point
                                    cv2.putText(
                                        filtered_img,
                                        shape,
                                        (obj_cx - 10, obj_cy + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )

                                    # Display X/Y position of the object center
                                    cv2.putText(
                                        filtered_img,
                                        f"X:{obj_cx} Y:{obj_cy}",
                                        (obj_cx - 10, obj_cy + 45),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (255, 255, 255),
                                        1,
                                        cv2.LINE_AA,
                                    )

                                    # Display distance text near the object
                                    dist_text = f"{total_dist_mm:.1f} mm"
                                    cv2.putText(
                                        filtered_img,
                                        dist_text,
                                        (obj_cx + 10, obj_cy - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (255, 255, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )

                        text = f"Ratio: {pixel_cm_ratio:.4f} px/mm"
                        cv2.putText(
                            filtered_img,
                            text,
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.putText(
                            filtered_img,
                            "No ArUco Marker Detected",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    cv2.imshow("Basler Camera Feed", filtered_img)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except Exception as e:
            print(f"Error caught: {e}")
            continue

    camera.StopGrabbing()
    cv2.destroyAllWindows()