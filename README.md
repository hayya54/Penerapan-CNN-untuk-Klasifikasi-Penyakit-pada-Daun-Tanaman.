# Penerapan-CNN-untuk-Klasifikasi-Penyakit-pada-Daun-Tanaman.

## 1. Pemahaman Bisnis
Tujuan utama dari proyek ini adalah untuk mengembangkan model yang dapat secara otomatis mengidentifikasi dan mengklasifikasikan penyakit pada daun tanaman berdasarkan citra gambar. Ini sangat relevan dalam bidang pertanian untuk membantu petani mendeteksi penyakit lebih awal, sehingga dapat mengambil tindakan pencegahan atau pengobatan yang tepat untuk menjaga kesehatan tanaman dan hasil panen.

## 2. Dataset
Dataset gambar daun tanaman disimpan di Google Drive Anda, dengan struktur folder:
2.1 train: Berisi gambar-gambar untuk pelatihan model.
2.2 valid: Berisi gambar-gambar untuk validasi model (meskipun dalam kode validasi diambil dari subset train_dir).
2.3 test: Berisi gambar-gambar untuk pengujian model.

Jumlah Gambar dan Kelas:
2.1 Data Training: Ditemukan 360 gambar yang termasuk dalam 4 kelas berbeda.
2.2 Data Validasi: Ditemukan 40 gambar yang termasuk dalam 4 kelas berbeda (diambil dari 10% subset data training).
2.3 Data Testing: Ditemukan 589 gambar yang termasuk dalam 4 kelas berbeda.

Setiap gambar diubah ukurannya menjadi 224x224 piksel (s=224) dan menggunakan mode warna RGB.

## 3. Pra-pemrosesan dan Augmentasi Data
Untuk mempersiapkan gambar dan meningkatkan variasi dataset, digunakan tf.keras.preprocessing.image.ImageDataGenerator dengan teknik-teknik berikut:
3.1 Normalisasi Piksel: Nilai piksel gambar direscale dari rentang 0-255 menjadi 0-1 (rescale=1/255.0).
3.2 Pembagian Data Validasi: 10% dari data training digunakan sebagai data validasi (validation_split=0.1).
3.4 Augmentasi Data (hanya untuk data training):
3.5 rotation_range=20: Rotasi gambar hingga 20 derajat.
3.6 width_shift_range=0.2: Pergeseran horizontal gambar.
3.7 height_shift_range=0.2: Pergeseran vertikal gambar.
3.8 shear_range=0.2: Transformasi shear.
3.9 zoom_range=0.2: Zoom in/out gambar.
3.10 horizontal_flip=True: Flip horizontal gambar.
3.11 brightness_range=[0.8, 1.2]: Penyesuaian kecerahan gambar.
3.12 Ukuran Batch: Gambar diproses dalam batch berukuran 16.
3.13 Mode Kelas: class_mode='categorical' digunakan karena ini adalah masalah klasifikasi multi-kelas.

## 4. Arsitektur Model CNN
Model yang dibangun adalah model Sequential CNN yang terdiri dari beberapa lapisan (layer):
4.1 Lapisan Konvolusi (Conv2D):
4.1.1 Lapisan pertama: 32 filter dengan ukuran kernel (3,3), fungsi aktivasi ReLU, dan input shape (224, 224, 3).
4.1.2 Lapisan kedua: 64 filter dengan ukuran kernel (3,3), fungsi aktivasi ReLU.
4.1.3 Lapisan Pooling (MaxPooling2D): Digunakan setelah setiap lapisan konvolusi dengan pool_size=(2,2) untuk mengurangi dimensi spasial.
4.1.4 Lapisan Dropout: Digunakan setelah setiap lapisan konvolusi dan sebelum lapisan Dense kedua dengan dropout(0.25) dan dropout(0.5) untuk mencegah overfitting.
4.1.5 Lapisan Flatten: Mengubah output dari lapisan konvolusi dan pooling menjadi vektor satu dimensi.
4.1.6 Lapisan Dense (Fully Connected):
4.1.7 Lapisan pertama: 128 unit (neuron) dengan fungsi aktivasi ReLU.
4.1.8 Lapisan output: num_classes (jumlah kelas, yaitu 4) unit dengan fungsi aktivasi softmax untuk klasifikasi multi-kelas.

Model ini memiliki total 23,907,908 parameter yang semuanya trainable (dapat dilatih).

## 5. Kompilasi dan Pelatihan Model
a. Optimizer: Model dikompilasi menggunakan optimizer 'adam'.
b. Fungsi Loss: Digunakan 'categorical_crossentropy' sebagai fungsi loss, yang sesuai untuk masalah klasifikasi multi-kelas dengan label one-hot encoded.
c. Metrik Evaluasi: Metrik 'accuracy' digunakan untuk memantau kinerja model selama pelatihan.
d. Pelatihan: Model dilatih selama 20 epoch menggunakan train_generator dan divalidasi dengan valid_generator. Parameter steps_per_epoch dan validation_steps dihitung berdasarkan jumlah sampel dan ukuran batch.

Selama pelatihan, akurasi pada data training dan validasi dipantau, contoh output menunjukkan peningkatan akurasi dari waktu ke waktu.

## 6. Evaluasi Model
Model dievaluasi pada data uji (test_generator).
Akurasi pada Data Uji: Akurasi yang diperoleh adalah 64.86%.
Kode juga menyertakan fungsi untuk visualisasi hasil prediksi:
a. plot_image: Menampilkan gambar beserta label prediksi, probabilitas prediksi, dan label sebenarnya. Warna teks prediksi akan biru jika benar dan merah jika salah.
b. plot_value_array: Menampilkan bar chart probabilitas prediksi untuk setiap kelas, dengan bar yang diprediksi berwarna merah dan bar label sebenarnya berwarna biru.
Beberapa contoh gambar dan prediksinya juga ditampilkan untuk menunjukkan kinerja model secara visual.

7. Library yang Digunakan
Berikut adalah library Python yang diimpor untuk analisis ini:
7.1 matplotlib.pyplot: Untuk membuat visualisasi (misalnya, plot gambar dan bar chart).
7.2 tensorflow: Kerangka kerja utama untuk membangun dan melatih model CNN.
7.3 numpy: Untuk operasi numerik.
7.4 tensorflow.keras.models.Sequential: Untuk membangun model secara sekuensial (lapisan demi lapisan).
7.5 tensorflow.keras.layers.Conv2D: Untuk lapisan konvolusi.
7.6 tensorflow.keras.layers.MaxPooling2D: Untuk lapisan pooling.
7.7 tensorflow.keras.layers.Flatten: Untuk meratakan tensor.
7.8 tensorflow.keras.layers.Dense: Untuk lapisan fully connected.
7.9 tensorflow.keras.layers.Dropout: Untuk lapisan dropout.
7.10 tensorflow.keras.preprocessing.image.ImageDataGenerator: Untuk pra-pemrosesan dan augmentasi gambar.
