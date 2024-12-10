import qrcode

# QRコード情報設定
joint_number = 0
bolt_count = 50
neck_length = 20
bolt_type = "A"

data = {
    "添接番号": joint_number,
    "ボルト本数": bolt_count,
    "首下長さ": neck_length,
    "種類": bolt_type,
}

encoded_data = str(data)

# QRコード生成
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=30,
    border=4,
)
qr.add_data(encoded_data)
qr.make(fit=True)


# QRコード画像生成
img = qr.make_image(fill_color="black", back_color="white")

# QRコード画像保存
img.save("qr_code_big.png")
