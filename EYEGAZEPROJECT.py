import time

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Daftar 5 Soal Matematika beserta Kunci Jawabannya
DAFTAR_SOAL = [
    {
        "soal": "Berapakah hasil dari 3 + 4 ?",
        "opsi": ["A. 5", "B. 7", "C. 10", "D. 12", "E. 15"],
        "kunci": "B",
    },
    {
        "soal": "Berapakah hasil dari 12 - 5 ?",
        "opsi": ["A. 6", "B. 8", "C. 7", "D. 9", "E. 10"],
        "kunci": "C",
    },
    {
        "soal": "Berapakah hasil dari 6 x 3 ?",
        "opsi": ["A. 12", "B. 16", "C. 20", "D. 18", "E. 24"],
        "kunci": "D",
    },
    {
        "soal": "Berapakah hasil dari 20 / 4 ?",
        "opsi": ["A. 4", "B. 5", "C. 6", "D. 7", "E. 8"],
        "kunci": "B",
    },
    {
        "soal": "Berapakah hasil dari 7 + 8 ?",
        "opsi": ["A. 13", "B. 14", "C. 15", "D. 16", "E. 17"],
        "kunci": "C",
    },
]

RAW_LABELS = ["A", "B", "C", "D", "E"]
BOX_W, BOX_H = 150, 90
BOX_MARGIN = 20
EAR_THRESHOLD = 0.22
DWELL_BLINK_TIME = 3.0


def hitung_ear(face_landmarks, frame_w, frame_h):
  try:
    l_top = face_landmarks.landmark[159]
    l_bot = face_landmarks.landmark[145]
    l_out = face_landmarks.landmark[33]
    l_in = face_landmarks.landmark[133]

    r_top = face_landmarks.landmark[386]
    r_bot = face_landmarks.landmark[374]
    r_out = face_landmarks.landmark[263]
    r_in = face_landmarks.landmark[362]

    v_kiri = abs(l_top.y - l_bot.y) * frame_h
    h_kiri = abs(l_in.x - l_out.x) * frame_w
    ear_kiri = v_kiri / h_kiri if h_kiri > 0 else 0

    v_kanan = abs(r_top.y - r_bot.y) * frame_h
    h_kanan = abs(r_in.x - r_out.x) * frame_w
    ear_kanan = v_kanan / h_kanan if h_kanan > 0 else 0

    return (ear_kiri + ear_kanan) / 2.0
  except Exception:
    return 0.3


def deteksi_arah_lirikan_langsung(face_landmarks, frame_w, frame_h):
  try:
    l_out = face_landmarks.landmark[33]
    l_in = face_landmarks.landmark[133]
    l_pupil = face_landmarks.landmark[468]

    r_out = face_landmarks.landmark[263]
    r_in = face_landmarks.landmark[362]
    r_pupil = face_landmarks.landmark[473]

    def to_pixel(lm):
      return int(lm.x * frame_w), int(lm.y * frame_h)

    pts = {
        "l_out": to_pixel(l_out),
        "l_in": to_pixel(l_in),
        "l_pupil": to_pixel(l_pupil),
        "r_out": to_pixel(r_out),
        "r_in": to_pixel(r_in),
        "r_pupil": to_pixel(r_pupil),
    }

    l_center_x = (l_out.x + l_in.x) / 2.0
    r_center_x = (r_out.x + r_in.x) / 2.0

    selisih_kiri = l_pupil.x - l_center_x
    selisih_kanan = r_pupil.x - r_center_x
    rata_selisih = (selisih_kiri + selisih_kanan) / 2.0

    return rata_selisih, pts
  except Exception:
    return 0.0, None


def gambar_ui(
    image,
    indeks_soal,
    indeks_fokus,
    jawaban_terkunci_sementara,
    status_jawaban_teks,
    progres_kedip,
    landmark_mata,
    arah_val,
    selesai,
    skor,
):
  frame_h, frame_w, _ = image.shape

  if landmark_mata:
    cv2.circle(image, landmark_mata["l_out"], 4, (255, 0, 0), -1)
    cv2.circle(image, landmark_mata["l_in"], 4, (255, 0, 0), -1)
    cv2.circle(image, landmark_mata["r_out"], 4, (255, 0, 0), -1)
    cv2.circle(image, landmark_mata["r_in"], 4, (255, 0, 0), -1)
    cv2.circle(image, landmark_mata["l_pupil"], 6, (0, 0, 255), -1)
    cv2.circle(image, landmark_mata["r_pupil"], 6, (0, 0, 255), -1)

  if selesai:
    # Tampilan Selesai Ujian
    cv2.putText(
        image,
        "UJIAN SELESAI!",
        (frame_w // 2 - 180, frame_h // 2 - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3,
    )
    cv2.putText(
        image,
        f"Skor Akhir Anda: {skor} dari {len(DAFTAR_SOAL)}",
        (frame_w // 2 - 210, frame_h // 2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        "Tekan ESC untuk Keluar",
        (frame_w // 2 - 140, frame_h // 2 + 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
    )
    return

  # Render Soal Aktif
  soal_saat_ini = DAFTAR_SOAL[indeks_soal]
  cv2.putText(
      image,
      f"Soal {indeks_soal + 1} dari {len(DAFTAR_SOAL)}",
      (30, 130),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.9,
      (0, 255, 255),
      2,
  )
  cv2.putText(
      image,
      soal_saat_ini["soal"],
      (30, 180),
      cv2.FONT_HERSHEY_SIMPLEX,
      1.2,
      (255, 255, 255),
      3,
  )

  # Render Opsi Jawaban
  total_w = (
      len(soal_saat_ini["opsi"]) * BOX_W
      + (len(soal_saat_ini["opsi"]) - 1) * BOX_MARGIN
  )
  start_x = (frame_w - total_w) // 2
  box_y = frame_h - 160

  for i, label in enumerate(soal_saat_ini["opsi"]):
    x1 = start_x + i * (BOX_W + BOX_MARGIN)
    y1 = box_y
    x2, y2 = x1 + BOX_W, y1 + BOX_H

    if RAW_LABELS[i] == jawaban_terkunci_sementara:
      warna_isi = (0, 200, 0)
    elif i == indeks_fokus:
      warna_isi = (0, 255, 255)
    else:
      warna_isi = (50, 50, 50)

    cv2.rectangle(image, (x1, y1), (x2, y2), warna_isi, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    ukuran_teks = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    tx = x1 + (BOX_W - ukuran_teks[0]) // 2
    ty = y1 + (BOX_H + ukuran_teks[1]) // 2
    cv2.putText(
        image,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

  cv2.putText(
      image,
      f"Offset Mata: {arah_val:.4f}",
      (30, 90),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (0, 0, 255),
      2,
  )

  if status_jawaban_teks:
    # Tampilkan notifikasi benar/salah saat jeda
    cv2.putText(
        image,
        status_jawaban_teks,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if "BENAR" in status_jawaban_teks else (0, 0, 255),
        3,
    )
  else:
    persen_kedip = int(progres_kedip * 100)
    cv2.putText(
        image,
        (
            f"Fokus: {RAW_LABELS[indeks_fokus]} | Tahan Kedip 3s untuk Pilih:"
            f" {persen_kedip}%"
        ),
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def main():
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  indeks_soal = 0
  indeks_fokus = 0
  skor = 0
  selesai = False
  jawaban_terkunci_sementara = None
  status_jawaban_teks = ""

  # Variabel untuk mengontrol jeda (delay) setelah jawab
  waktu_jeda_mulai = None
  JEDA_DURASI = 2.0  # Jeda selama 2 detik

  sedang_melirik = False
  AMBANG_BATAS = 0.012

  waktu_mulai_kedip = None
  progres_kedip = 0.0

  window_name = "Exam - Math Quiz with Delay"
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
  ) as face_mesh:

    while cap.isOpened():
      success, image = cap.read()
      if not success:
        break

      image = cv2.flip(image, 1)
      frame_h, frame_w, _ = image.shape
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(image_rgb)

      landmark_mata_data = None
      arah_val = 0.0

      if results.multi_face_landmarks and not selesai:
        face_landmarks = results.multi_face_landmarks[0]
        waktu_sekarang = time.time()

        # Jika sedang dalam masa jeda setelah menjawab, abaikan input mata/kedip sementara
        if waktu_jeda_mulai is not None:
          if waktu_sekarang - waktu_jeda_mulai >= JEDA_DURASI:
            waktu_jeda_mulai = None
            jawaban_terkunci_sementara = None
            status_jawaban_teks = ""
            indeks_soal += 1
            indeks_fokus = 0
            if indeks_soal >= len(DAFTAR_SOAL):
              selesai = True
        else:
          ear = hitung_ear(face_landmarks, frame_w, frame_h)
          sedang_kedip = ear < EAR_THRESHOLD

          arah_val, landmark_mata_data = deteksi_arah_lirikan_langsung(
              face_landmarks, frame_w, frame_h
          )

          if not sedang_kedip:
            if not sedang_melirik:
              if arah_val < -AMBANG_BATAS:
                indeks_fokus = (indeks_fokus - 1) % len(
                    DAFTAR_SOAL[indeks_soal]["opsi"]
                )
                sedang_melirik = True
              elif arah_val > AMBANG_BATAS:
                indeks_fokus = (indeks_fokus + 1) % len(
                    DAFTAR_SOAL[indeks_soal]["opsi"]
                )
                sedang_melirik = True
            else:
              if abs(arah_val) < (AMBANG_BATAS / 2):
                sedang_melirik = False

          if sedang_kedip:
            if waktu_mulai_kedip is None:
              waktu_mulai_kedip = waktu_sekarang
            durasi_kedip = waktu_sekarang - waktu_mulai_kedip
            progres_kedip = min(durasi_kedip / DWELL_BLINK_TIME, 1.0)

            if durasi_kedip >= DWELL_BLINK_TIME:
              pilihan_user = RAW_LABELS[indeks_fokus]
              kunci_jawaban = DAFTAR_SOAL[indeks_soal]["kunci"]
              jawaban_terkunci_sementara = pilihan_user

              if pilihan_user == kunci_jawaban:
                skor += 1
                status_jawaban_teks = f"Jawaban {pilihan_user}: BENAR!"
              else:
                status_jawaban_teks = (
                    f"Jawaban {pilihan_user}: SALAH! (Kunci: {kunci_jawaban})"
                )

              # Aktifkan waktu jeda
              waktu_jeda_mulai = time.time()
              waktu_mulai_kedip = None
              progres_kedip = 0.0
          else:
            waktu_mulai_kedip = None
            progres_kedip = 0.0

      gambar_ui(
          image,
          indeks_soal,
          indeks_fokus,
          jawaban_terkunci_sementara,
          status_jawaban_teks,
          progres_kedip,
          landmark_mata_data,
          arah_val,
          selesai,
          skor,
      )
      cv2.imshow(window_name, image)

      if cv2.waitKey(5) & 0xFF == 27:
        break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
