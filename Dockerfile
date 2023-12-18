# Sử dụng hình ảnh Python 3.9
FROM python:3.10

# Tạo thư mục làm việc và đặt nó làm thư mục làm việc mặc định
WORKDIR /app

# Sao chép tất cả các tệp yêu cầu vào thư mục làm việc
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app/
# Expose cổng cho ứng dụng Flask
EXPOSE 8899
# Khởi chạy ứng dụng Flask
CMD ["python", "mapmatching.py"]