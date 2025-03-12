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
pip install -r requirements.txt
```

## 3. อัปเดต Pip และตรวจสอบการติดตั้ง

```sh
pip install --upgrade pip
pip list
```
## 4. ดาวน์โหลด extension ใน vscode
โหลด jupyter และ python จาก extension
## 5. วิธีใช้งานไฟล์ `train_model.ipynb`

ไฟล์นี้ใช้สำหรับการฝึกโมเดลวัตถุด้วย YOLO และบันทึกโมเดลที่ได้

### 5.1 โครงสร้างหลักของ `train_model.ipynb`
1. **เลือก kernel ในไฟล์ให้เป็นenviroment ที่เราได้ติดตั้งไว้ในข้อ 1-3**
2. **นำเข้าไลบรารีทั้งหมดจากใน cell ที่กำหนดไว้**
3. **โหลดและเตรียมข้อมูล** - แตกไฟล์ zip ที่กำหนดไว้เป็นdataset หรือนำเข้าข้อมูลที่ต้องการเทรนโดย format COCO
4. **ตั้งค่า YOLO Model** - โหลดโครงสร้าง `yolov12n.yaml` และกำหนดค่าต่าง ๆ
5. **ฝึกโมเดล (Training)** - ใช้ `model.train(...)` เพื่อเทรนโมเดลบนข้อมูลที่มี
6. **บันทึกโมเดลที่ดีที่สุด** - ไฟล์โมเดลที่ดีที่สุดจะถูกบันทึกที่ `runs/detect/train/weights/best.pt`

### 4.2 วิธีการใช้งาน
1. เปิดไฟล์ `train_model.ipynb` ด้
2. รันเซลล์แต่ละเซลล์ตามลำดับ เพื่อฝึกโมเดล
3. ตรวจสอบผลลัพธ์และบันทึกโมเดล

## 6. วิธีใช้งานไฟล์ `test_model.ipynb`

ไฟล์นี้ใช้สำหรับทดสอบโมเดลที่ฝึกแล้ว

### 6.1 วิธีการใช้งาน
1. เปิดไฟล์ `test_model.ipynb`
2. เลือก kernel ในไฟล์ให้เป็นenviroment ที่เราได้ติดตั้งไว้ในข้อ 1-3
3. นำโมเดลที่ฝึกแล้วมาใส่ในโฟเดอร์ test_model ('best.pt')
4. อัปโหลดภาพที่ต้องการทดสอบ
5. ระบบจะแสดงผลลัพธ์การทำนายจากโมเดลที่ฝึกแล้ว

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

