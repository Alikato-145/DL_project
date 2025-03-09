# คู่มือการติดตั้งและการใช้งานสำหรับ Anaconda

ไฟล์นี้เป็น `README.md` ที่ใช้สำหรับแนะนำการติดตั้งและใช้งานโมเดลใน Anaconda รวมถึงการนำไปใช้งานใน GitHub

## 1. สร้างและเปิดใช้งาน Virtual Environment

```sh
conda create --name myenv python=3.9 -y
conda activate myenv
```

## 2. ติดตั้งไลบรารีที่จำเป็น

### ติดตั้งผ่าน Conda

```sh
conda install -c conda-forge numpy pillow tensorflow -y
conda install -c conda-forge opencv -y
conda install -c pytorch torchvision -y
```

### ติดตั้งผ่าน Pip

```sh
pip install ipython colorsys
```

## 3. อัปเดต Pip และตรวจสอบการติดตั้ง

```sh
pip install --upgrade pip
pip list
```

## 4. วิธีใช้งานไฟล์ `train_model.ipynb`

ไฟล์นี้ใช้สำหรับการฝึกโมเดลวัตถุด้วย YOLO และบันทึกโมเดลที่ได้

### 4.1 โครงสร้างหลักของ `train_model.ipynb`
1. **นำเข้าไลบรารีที่จำเป็น** เช่น `torch`, `tensorflow`, `cv2`, `PIL`, และ `ultralytics`
2. **โหลดและเตรียมข้อมูล** - ดึงชุดข้อมูลจาก Roboflow หรือโฟลเดอร์ที่กำหนด
3. **ตั้งค่า YOLO Model** - โหลดโครงสร้าง `yolov12n.yaml` และกำหนดค่าต่าง ๆ
4. **ฝึกโมเดล (Training)** - ใช้ `model.train(...)` เพื่อเทรนโมเดลบนข้อมูลที่มี
5. **บันทึกโมเดลที่ดีที่สุด** - ไฟล์โมเดลที่ดีที่สุดจะถูกบันทึกที่ `runs/detect/train/weights/best.pt`

### 4.2 วิธีการใช้งาน
1. เปิดไฟล์ `train_model.ipynb` ด้วย Jupyter Notebook หรือ Jupyter Lab
2. รันเซลล์แต่ละเซลล์ตามลำดับ เพื่อฝึกโมเดล
3. ตรวจสอบผลลัพธ์และบันทึกโมเดล

## 5. วิธีใช้งานไฟล์ `test.ipynb`

ไฟล์นี้ใช้สำหรับทดสอบโมเดลที่ฝึกแล้ว

### 5.1 วิธีการใช้งาน
1. เปิดไฟล์ `test.ipynb`
2. นำโมเดลที่ฝึกแล้วมาใส่ในโฟเดอร์ test model
3. อัปโหลดภาพที่ต้องการทดสอบ
4. ระบบจะแสดงผลลัพธ์การทำนายจากโมเดลที่ฝึกแล้ว

## 6. การใช้งานบน GitHub

หากต้องการใช้งานบน GitHub ให้ทำตามขั้นตอนนี้:

```sh
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <URL_REPO>
git push -u origin main
```

หลังจาก Push แล้ว ไฟล์ `README.md` จะแสดงผลอัตโนมัติบนหน้าโครงการของ GitHub 🚀

