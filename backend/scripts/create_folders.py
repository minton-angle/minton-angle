"""
data 폴더 내 하위 폴더만 생성
"""
import os

# 생성할 하위 폴더
subfolders = [
    "upload",
    "upload_keyframes",
    "standard",
    "temp",
]

print("=" * 60)
print("📁 하위 폴더 생성 중...")
print("=" * 60)

# 경로 설정
current_file = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file)
backend_dir = os.path.dirname(script_dir)
data_dir = os.path.join(backend_dir, "data")

print(f"📍 Data 폴더: {data_dir}\n")

# data 폴더 존재 확인
if not os.path.exists(data_dir):
    print(f"❌ data 폴더가 없습니다!")
    exit(1)

# 하위 폴더 생성
for folder in subfolders:
    folder_path = os.path.join(data_dir, folder)
    
    try:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"✅ 생성: {folder}/")
        else:
            print(f"⏭️  존재: {folder}/")
    except Exception as e:
        print(f"❌ 에러 ({folder}): {e}")

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)

# 최종 확인
print("\n📂 data 폴더 내용:")
for item in sorted(os.listdir(data_dir)):
    item_path = os.path.join(data_dir, item)
    if os.path.isdir(item_path):
        print(f"   📁 {item}/")
    else:
        print(f"   📄 {item}")



# """
# data 폴더 내 하위 폴더만 생성 (디버깅)
# """
# import os

# print("=" * 60)
# print("🔍 경로 확인")
# print("=" * 60)

# # 1. 현재 파일 위치
# current_file = os.path.abspath(__file__)
# print(f"1️⃣ 현재 파일: {current_file}")

# # 2. scripts 폴더
# script_dir = os.path.dirname(current_file)
# print(f"2️⃣ scripts 폴더: {script_dir}")

# # 3. backend 폴더
# backend_dir = os.path.dirname(script_dir)
# print(f"3️⃣ backend 폴더: {backend_dir}")

# # 4. data 폴더
# data_dir = os.path.join(backend_dir, "data")
# print(f"4️⃣ data 폴더: {data_dir}")

# # 5. data 폴더 존재 확인
# print(f"\n📂 data 폴더 존재? {os.path.exists(data_dir)}")

# if os.path.exists(data_dir):
#     print(f"📂 data 폴더 내용:")
#     for item in os.listdir(data_dir):
#         print(f"   - {item}")

# print("\n" + "=" * 60)
# print("일단 여기까지만 실행!")
# print("=" * 60)