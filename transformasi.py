from datetime import datetime
import pandas as pd

def transformasi(df):
    ##Penghitungan Umur Usaha
    for index, row in df.iterrows():
        df.loc[index, 'Umur Usaha'] = datetime.now().year - int ( row['Tgl Pendirian Usaha'][-4:])

    df1 = df['Pendidikan'].str.get_dummies(sep=',')

    df2 = df['Kegiatan Usaha'].str.get_dummies(sep=', ')

    df3 = df['Tujuan Pemasaran'].str.get_dummies(sep=', ')

    df4 = df['Status Kepemilikan Tanah'].str.get_dummies(sep=',')

    df5 = df['Sarana Media Elektronik'].str.get_dummies(sep=', ')

    df6 = df['Modal Bantuan Pemerintah'].str.get_dummies(sep=',')

    df7 = df['Pinjaman'].str.get_dummies(sep=', ')

    df8 = df['Omset Pertahun'].str.get_dummies(sep=',')

    df9 = df['Kepemilikan Asuransi Kesehatan'].str.get_dummies(sep=', ')

    newdf = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df['Tenaga Kerja L'], df['Tenaga Kerja P'], df['Umur Usaha']], axis='columns')

    return newdf