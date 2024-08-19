import os
from pypylon import pylon
import cv2

# Đường dẫn tới thư mục lưu trữ ảnh
save_folder = 'data/qr_code/input'

# Kiểm tra và tạo thư mục nếu chưa tồn tại
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Khởi tạo một đối tượng Camera của Basler
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

try:
    # Kết nối camera
    camera.Open()

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Bắt đầu acquisition (chụp ảnh)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Biến đếm số lượng ảnh đã chụp
    image_count = 21

    while camera.IsGrabbing():
        # Lấy dữ liệu hình ảnh từ camera
        grabResult = camera.RetrieveResult(4000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Chuyển đổi thành ảnh OpenCV
            image = grabResult.Array

            img_resized = cv2.resize(image, (480,560))

            # Hiển thị khung hình
            cv2.imshow("Camera Stream", img_resized)

            # Kiểm tra sự kiện nhấn phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Tạo tên file để lưu
                filename = os.path.join(save_folder, f"img_{image_count}.jpg")

                # Lưu ảnh vào file
                cv2.imwrite(filename, img_resized)
                print(f"Đã lưu {filename}")

                image_count += 1

        grabResult.Release()

    # Dừng acquisition và giải phóng camera
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Lỗi: {e}")

finally:
    # Đảm bảo rằng camera được giải phóng ngay cả khi có lỗi xảy ra
    if camera.IsOpen():
        camera.Close()