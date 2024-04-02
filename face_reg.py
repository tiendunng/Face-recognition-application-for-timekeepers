import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
import datetime

#Đây là hàm để chỉnh lại kích thước của ảnh
def conditional_resize(img, mode = 0):
    img_width, img_height, channels = img.shape

    if mode == 0:
        if(img_width >= 2000 or img_height >= 2000):
            resized_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
        else:
            resized_img = img
    elif mode == 1:
        resized_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    return resized_img

#Đây là hàm để chuyển dạng từ dạng mà opencv có thể xử lý sang dạng face_recognition có thể xử lý
def Cv_To_Face_Reg(img):

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_face_reg = np.array(rgb_img)

    return img_face_reg

#Hàm để lấy các loại hình
def read_imgs(imgs_folder, img_types):
    known_faces = []
    known_names = []
    for filename in os.listdir(imgs_folder):
        #Chỉ lấy file có định dạng trong img_types
        if os.path.splitext(filename)[1] in img_types:
            folder_name = os.path.basename(imgs_folder)
            path_to_img = os.getcwd() + "\\" + folder_name + "\\" + filename
            
            #Đọc file ảnh
            main_img = cv2.imread(path_to_img)

            #Chỉnh lại kích thước ảnh vì face_recognition chỉ có thể hoạt động ở mức kích thước nhất định
            resized_img = conditional_resize(main_img)
        
            #Chuyển định dạng
            img = Cv_To_Face_Reg(resized_img)

            #Trích xuất đặt tính của khuôn mặt
            face_encoding = face_recognition.face_encodings(img)[0]

            #Gán các giá trị chiết xuất được
            known_faces.append(face_encoding)
            known_names.append(filename.split(".")[0])

        else:
            continue
    return known_faces, known_names

#Hàm lấy thời điểm ca làm trong ngày dựa vào mốc thời gian 12h
def get_shift():
    current_time = datetime.datetime.now().time()
    current_hour = current_time.hour
    if int(current_hour) < 12:
        sh_name = "Morning"
    else:
        sh_name = "Evening"
    
    return sh_name

#Hàm lấy dataset file excel trong đường folder 
def get_df(path_to_folder, sh_name, column_names, dt_needed):
    
    current_date = datetime.date.today()
    current_date = current_date.strftime("%d/%m/%Y")

    #Đặt tên cho file bằng ngày tháng năm sẽ không bị trùng
    excel_name = current_date.replace("/", "_") + ".xlsx"
    excel_file_path = path_to_folder + "\\" + excel_name
    
    #Kiểm tra hiện diện nếu file dã hiện diện thì lấy dữ liệu
    #ngược lại thì tạo file mới
    file_exist = False
    for filename in os.listdir(path_to_folder):
        if filename == excel_name:
            file_exist = True

            df = pd.read_excel(excel_file_path, sh_name)
            dt = df[dt_needed]
            
            return dt, excel_file_path
        
    if not file_exist:
        placeholder_sheet = pd.DataFrame(columns = column_names)

        with pd.ExcelWriter(excel_file_path, engine="xlsxwriter") as writer:
            placeholder_sheet.to_excel(writer, sheet_name="Morning", index=False)
            placeholder_sheet.to_excel(writer, sheet_name="Evening", index=False)

        placeholding = placeholder_sheet[dt_needed]
        return placeholding, excel_file_path

    return []

def loopCam(imgs_folder, attendanceFolder, dtsFile, img_types, column_names, dt_needed, updatedf = True):

    #Lấy đặt tính các khuôn mặt và tên tương ứng
    known_faces, known_id = read_imgs(imgs_folder, img_types)

    #Chọn camera (mặc định là 0)
    cap = cv2.VideoCapture(0)
    while True:
        #Lấy từng frame của camera
        ret, frame = cap.read()
        shift = get_shift()
        #Dùng If với updatedf để tránh việc đọc lại file df quá nhiều lần
        if updatedf:
            df, excel_path = get_df(attendanceFolder, shift, column_names, dt_needed)
            
            dts = pd.read_excel(dtsFile, sheet_name="Sheet1")


            updatedf = False


        #Xoay ảnh để hợp với hành động của người ngoài ảnh
        frame = cv2.flip(frame,1)

        #Sửa kích thước để chạy nhanh hơn và chuyển dạng
        small_frame = conditional_resize(frame,mode = 1)
        face_reg_img = Cv_To_Face_Reg(small_frame)

        #Trích xuất thông tin vị trí và đặt tính khuôn mặt
        #(mỗi face_locations gồm nhiều array và mỗi array chứa 4 tọa đóng khung khuôn mặt)
        face_locations = face_recognition.face_locations(face_reg_img)
        face_encodings = face_recognition.face_encodings(face_reg_img, face_locations)
        
        #Array để lưu lại các thông tin của từng người đang hiện trên camera
        #Đồng thời dùng để tránh thêm trùng lặp vào file excel
        confirmed_individuals = []

        font = cv2.FONT_HERSHEY_PLAIN

        cv2.putText(frame, "Current Shift: " + shift,(10, 40), font, 1.2, (0, 0, 0), 1 )

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            #Chỉnh lại vị trí các tọa độ cho khớp với hình ban đầu
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            #So sánh đặt điểm tất cả khuôn mặt có trước so với khuôn mặt đang có
            matches = face_recognition.compare_faces(known_faces, face_encoding)

            #Tìm giá trị bức hình có tỉ lệ đúng gần với hình ảnh đang có
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)

            #Lấy tên cho bức hình có giá trị đúng nhất
            if matches[best_match_index]:
                id_num = known_id[best_match_index]

            try:
                name = dts.loc[dts['ID'] == id_num, 'NAME'].values[0]
            except:
                name = "Unknown"

            #Viết tên và các thông tin liên quan
            target_name = "Name: %s" % name

            #Dựa vào các tên và trạng thái sẽ cho vẽ khác nhau
            #Không nhận được dạng: Đen
            #Nhận được dạng nhưng chưa chấm công: Đỏ
            #Nhận được dạng và dã chấm công: Xanh
            
            if name == "Unknown":
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
                cv2.putText(frame, name, (left + 5 , bottom + 20), font, 1.0, (0, 0, 0), 1)
            else:
                target_name = "Name: %s" % name
                target_id = "Id: %s" % id_num
                stat = "Not checked"

                exist_id = id_num in df["Id"].values
                
                if exist_id:
                    stat = df.loc[df["Id"] == id_num, "Attendance Status"].values[0]

                for x in [id_num]:
                    confirmed_individuals.append(x)


                target_stat = "Status: %s" % stat

                if stat == "Checked":
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    cv2.putText(frame, target_name, (left + 5 , bottom + 20), font, 1.0, (0, 255, 0), 1)
                    cv2.putText(frame, target_id, (left + 5 , bottom + 35), font, 1.0, (0, 255, 0), 1)
                    cv2.putText(frame, target_stat, (left + 5 , bottom + 50), font, 1.0, (0, 255, 0), 1)

                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    cv2.putText(frame, target_name, (left + 5 , bottom + 20), font, 1.0, (0, 0, 255), 1)
                    cv2.putText(frame, target_id, (left + 5 , bottom + 35), font, 1.0, (0, 0, 255), 1)
                    cv2.putText(frame, target_stat, (left + 5 , bottom + 50), font, 1.0, (0, 0, 255), 1)
         

        #Thể hiện các hình
        cv2.imshow('Now Showing', frame)

        #Key y để đồng ý chấm công
        if cv2.waitKey(1) & 0xFF == ord('y'):
            current_time = datetime.datetime.now().time()
            format_time = "{:02d}:{:02d}:{:02d}".format(current_time.hour, current_time.minute, current_time.second)
            
            individual_list = []
            for x in confirmed_individuals:
                # if x[2] == "Checked":
                #     continue
                other_info = dts.loc[dts['ID'] == x].values[0][1::]
                individual_list.append([x,  other_info[0], other_info[1], other_info[2], "Checked", format_time])
            
            temp_df = pd.DataFrame(individual_list)

            with pd.ExcelWriter(excel_path, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
                temp_df.to_excel(writer,sheet_name=get_shift(), startrow=len(df) + 1, header=False, index=False)
            
            updatedf = True
        #Key q để kết thúc
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Tắt cam và các khung hình
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    attendanceFolder = r"Attendance_Status"
    imageFolder = r"Face_images"
    dtsFile = r"D:\DACNTT\Data\DS.xlsx"


    img_types = [".pngs", ".jpg", ".jpeg"]
    column_names = ["Id", "Name", "Department", "Position", "Attendance Status", "Last Update"]
    dt_needed = ["Id", "Attendance Status"]

    loopCam(imageFolder, attendanceFolder, dtsFile, img_types, column_names, dt_needed)