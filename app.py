from flask import Flask, render_template, request, session, url_for, redirect, flash
import pandas as pd
import numpy as np
import mysql.connector
import transformasi
from datetime import datetime
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'anitarahma'

db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='ukm_kuliner'
)

cursor = db.cursor()

ALLOWED_EXTENSIONS = {'xlsx'}

@app.route('/', methods=['GET', 'POST'])
def index():
    # menampilkan semua data
    cursor.execute("SELECT * FROM kuliner")
    result=cursor.fetchall()
    return render_template('index.html', result=result)

@app.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':

        # cek nama dan ekstensi file
        file = request.files['file']
        if file.filename == '':
            flash('tidak ada file yang diupload')
            return redirect(url_for('upload'))
        if file.filename.rsplit('.',1)[1].lower() not in ALLOWED_EXTENSIONS:
            flash('file harus dalam ekstensi xlsx (excel)')
            return redirect(url_for('upload'))
        
        df = pd.read_excel(file)

        # input data dari file ke database
        for index, row in df.iterrows():
            sql = "INSERT INTO kuliner (ref_oss, nik, nama, tgl_lahir, usia, jk, pendidikan, no_telp, email, provinsi, kab_kota, kecamatan, desa, nama_jln, nama_usaha, nib, tgl_terbit_nib, tgl_pendirian_usaha, koordinat, bidang_usaha, sektor_usaha, kegiatan_usaha, produk_komoditas_ekspor, tujuan_pemasaran, status_kepemilikan_tanah, sarana_media_elektronik, modal_bantuan_pemerintah, pinjaman, omset_pertahun, kepemilikan_asuransi_kesehatan, tenaga_kerja_l, tenaga_kerja_p, rerata_usia, status_formulir) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (row['Ref. OSS'], row['Nomor Induk Kependudukan'], row['Nama Lengkap'], row['Tanggal Lahir'], row['Usia'], row['Jenis Kelamin'],
            row['Pendidikan Terakhir'], row['No. Telepon'], row['e-Mail'], row['Provinsi'], row['Kab/Kota'], row['Kecamatan'], row['Desa/Kel, RT, RW'],
            row['Nama Jalan'], row['Nama Usaha'], row['NIB'], row['Tanggal Terbit NIB'], row['Tanggal Pendirian Usaha'], row['Koordinat'],
            row['Bidang Usaha'], row['Sektor Usaha'], row['Kegiatan Usaha'], row['Produk Komoditas Ekspor'], row['Tujuan Pemasaran'],
            row['Status Kepemilikan Tanah/Bangunan'], row['Sarana Media Elektronik'], row['Modal Bantuan Pemerintah'],
            row['Pinjaman Kredit Usaha Rakyat'], row['Omset per-Tahun'], row['Kepemilikan Asuransi Kesehatan'], row['Laki-laki'],
            row['Perempuan'], row['Rerata Usia Pekerja'], row['Status Formulir'])
            cursor.execute(sql, val)
        db.commit()
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        # ambil data dari database
        cursor.execute('SELECT * FROM kuliner')
        result = cursor.fetchall()

        # dijadikan dataframe
        dataCleaning = pd.DataFrame(result)

        # rename kolom
        dataCleaning = dataCleaning.rename(columns={0: 'No', 1: 'Ref OSS', 2: 'NIK', 3: 'Nama', 4: 'Tanggal Lahir', 5: 'Usia', 6: 'Jenis Kelamin',
        7: 'Pendidikan', 8: 'No Telp', 9: 'Email', 10: 'Provinsi', 11: 'Kab.Kota', 12: 'Kecamatan', 13: 'Desa', 14: 'Nama Jalan',
        15: 'Nama Usaha', 16: 'NIB', 17: 'Tgl Terbit NIB', 18: 'Tgl Pendirian Usaha', 19: 'Koordinat', 20: 'Bidang Usaha',
        21: 'Sektor Usaha', 22: 'Kegiatan Usaha', 23: 'Produk Komoditas Ekspor', 24: 'Tujuan Pemasaran', 25: 'Status Kepemilikan Tanah',
        26: 'Sarana Media Elektronik', 27: 'Modal Bantuan Pemerintah', 28: 'Pinjaman', 29: 'Omset Pertahun',
        30: 'Kepemilikan Asuransi Kesehatan', 31: 'Tenaga Kerja L', 32: 'Tenaga Kerja P', 33: 'Rerata Usia', 34: 'Status Formulir'})

        # cleaning
        dataCleaning = dataCleaning.dropna(axis=1, how='any')

        # Seleksi
        dataSeleksi = dataCleaning.copy()
        dataSeleksi = dataSeleksi.loc[:, ['Pendidikan','Tgl Pendirian Usaha','Kegiatan Usaha','Tujuan Pemasaran','Status Kepemilikan Tanah','Sarana Media Elektronik','Modal Bantuan Pemerintah','Pinjaman','Omset Pertahun','Kepemilikan Asuransi Kesehatan','Tenaga Kerja L','Tenaga Kerja P']]

        # Transformasi
        dataTmp = dataSeleksi.copy()
        dataTransformasi = transformasi.transformasi(dataTmp)
        
        return render_template('preprocessing.html', dataCleaningHead = dataCleaning.columns, dataCleaning=dataCleaning.to_numpy(), dataSeleksiHead = dataSeleksi.columns, dataSeleksi = dataSeleksi.to_numpy(), dataTransformasiHead = dataTransformasi.columns, dataTransformasi = dataTransformasi.to_numpy())
    return render_template('preprocessing.html')

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    if request.method == 'POST':
        global cluster
        cluster = int(request.form['cluster'])
        # ==============================
        cursor.execute('SELECT * FROM kuliner')
        result = cursor.fetchall()
        dataCleaning = pd.DataFrame(result)
        dataCleaning = dataCleaning.rename(columns={0: 'No', 1: 'Ref OSS', 2: 'NIK', 3: 'Nama', 4: 'Tanggal Lahir', 5: 'Usia', 6: 'Jenis Kelamin',
        7: 'Pendidikan', 8: 'No Telp', 9: 'Email', 10: 'Provinsi', 11: 'Kab.Kota', 12: 'Kecamatan', 13: 'Desa', 14: 'Nama Jalan',
        15: 'Nama Usaha', 16: 'NIB', 17: 'Tgl Terbit NIB', 18: 'Tgl Pendirian Usaha', 19: 'Koordinat', 20: 'Bidang Usaha',
        21: 'Sektor Usaha', 22: 'Kegiatan Usaha', 23: 'Produk Komoditas Ekspor', 24: 'Tujuan Pemasaran', 25: 'Status Kepemilikan Tanah',
        26: 'Sarana Media Elektronik', 27: 'Modal Bantuan Pemerintah', 28: 'Pinjaman', 29: 'Omset Pertahun',
        30: 'Kepemilikan Asuransi Kesehatan', 31: 'Tenaga Kerja L', 32: 'Tenaga Kerja P', 33: 'Rerata Usia', 34: 'Status Formulir'})

        # cleaning
        dataCleaning = dataCleaning.dropna(axis=1, how='any')

        # Seleksi
        dataSeleksi = dataCleaning.copy()
        dataSeleksi = dataSeleksi.loc[:, ['Pendidikan','Tgl Pendirian Usaha','Kegiatan Usaha','Tujuan Pemasaran','Status Kepemilikan Tanah','Sarana Media Elektronik','Modal Bantuan Pemerintah','Pinjaman','Omset Pertahun','Kepemilikan Asuransi Kesehatan','Tenaga Kerja L','Tenaga Kerja P']]

        # Transformasi
        dataTmp = dataSeleksi.copy()
        dataTransformasi = transformasi.transformasi(dataTmp)

        # ==============================

        Medoids = KMedoids(n_clusters=cluster, random_state=0, method='pam', init='k-medoids++')

        Medoids = Medoids.fit(dataTransformasi)

        hasil = np.ascontiguousarray(dataTransformasi)
        Medoids_label = Medoids.predict(hasil)
        silh_score = silhouette_score(hasil, Medoids_label)

        dataTransformasi['Cluster'] = Medoids.labels_

        for index, row in dataTransformasi.iterrows():
            if row['Cluster'] == 0:
                dataSeleksi.loc[index, 'Label'] = 'Cluster-0'
            elif row['Cluster'] == 1:
                dataSeleksi.loc[index, 'Label'] = 'Cluster-1'
            elif row['Cluster'] == 2:
                dataSeleksi.loc[index, 'Label'] = 'Cluster-2'
            elif row['Cluster'] == 3:
                dataSeleksi.loc[index, 'Label'] = 'Cluster-3'
            elif row['Cluster'] == 4:
                dataSeleksi.loc[index, 'Label'] = 'Cluster-4'

        dataSeleksi['Label'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
        plt.xlabel('Label', labelpad=14)
        plt.ylabel('Jumlah', labelpad=14)
        plt.savefig('static/assets/img/cluster {}.png'.format(cluster),dpi=100)
        plt.title("Jumlah Cluster", y=1.02)

        return render_template('clustering.html', cluster=cluster, score = silh_score, dataHead = dataSeleksi.columns, data = dataSeleksi.to_numpy())

    return render_template('clustering.html')

@app.route('/delete')
def delete():
    # hapus = "DELETE FROM kuliner"
    hapus = "TRUNCATE TABLE kuliner"
    cursor.execute(hapus)
    db.commit()
    return redirect(url_for('index'))

#menampilkan grafik cluster
@app.route('/display_image')
def display_image():
	return redirect(url_for('static', filename='assets/img/cluster {}.png'.format(cluster), code=307))

if __name__ == '__main__':
    app.run(debug=True)