from PIL import Image
from pyzbar.pyzbar import decode

def decode_qr_code(image_array):
    """
    Giải mã mã QR từ ảnh đầu vào.
    
    Args:
        image_array (numpy.ndarray): Ảnh đầu vào dưới dạng mảng numpy.
        
    Returns:
        str: Nội dung giải mã được từ mã QR.
    """
    # Chuyển đổi mảng numpy thành đối tượng PIL Image
    image = Image.fromarray(image_array)
    
    # Giải mã mã QR
    decoded_objects = decode(image)
    
    if decoded_objects:
        # Trả về nội dung của mã QR đầu tiên (nếu có)
        return decoded_objects[0].data.decode('utf-8')
    else:
        return "Không tìm thấy mã QR."

def save_decoded_text(text, output_path):
    """
    Lưu nội dung giải mã được vào file.
    
    Args:
        text (str): Nội dung giải mã được.
        output_path (str): Đường dẫn đến file để lưu nội dung.
    """
    with open(output_path, 'w') as file:
        file.write(text)
