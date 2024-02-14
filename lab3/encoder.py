import sys
import numpy
from PIL import Image
from scipy import fftpack
import huffmanEncode
from bitstream import BitStream
import math


zigzagOrder = numpy.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42,
                           49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])

def my_DCT(block):
    result = numpy.zeros_like(block, dtype=float)
    for u in range(8):
        for v in range(8):
            cu = 1.0 / numpy.sqrt(2) if u == 0 else 1.0
            cv = 1.0 / numpy.sqrt(2) if v == 0 else 1.0

            sum_val = 0.0
            for x in range(8):
                for y in range(8):
                    sum_val += block[x, y] * \
                        numpy.cos((2 * x + 1) * u * numpy.pi / 16) * \
                        numpy.cos((2 * y + 1) * v * numpy.pi / 16)

            result[u, v] = 0.25 * cu * cv * sum_val

    return result

def zigzac(rows=8, cols=8):
    result = []
    values = numpy.arange(64)
    arr = values.reshape((8, 8))
    for i in range(rows + cols - 1):
        if i % 2 == 0:
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                result.append(arr[j, i - j])
        else:
            for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                result.append(arr[j, i - j])
    return numpy.array(result).reshape([64])

def psnr(inputFile, outputFile):
    image = Image.open(inputFile)
    I = numpy.array(image)
    image = Image.open(outputFile)
    K = numpy.array(image)
    m, n = image.size

    MAX = 255
    temp = (I - K)**2
    MSE = numpy.sum(temp)/(m*n)
    result = 10*numpy.log10(MAX**2/MSE)
    return result

std_luminance_quant_tbl = numpy.array(
    [16,  11,  10,  16,  24,  40,  51,  61,
     12,  12,  14,  19,  26,  58,  60,  55,
     14,  13,  16,  24,  40,  57,  69,  56,
     14,  17,  22,  29,  51,  87,  80,  62,
     18,  22,  37,  56,  68, 109, 103,  77,
     24,  35,  55,  64,  81, 104, 113,  92,
     49,  64,  78,  87, 103, 121, 120, 101,
     72,  92,  95,  98, 112, 100, 103,  99], dtype=int)
std_luminance_quant_tbl = std_luminance_quant_tbl.reshape([8, 8])

std_chrominance_quant_tbl = numpy.array(
    [17,  18,  24,  47,  99,  99,  99,  99,
     18,  21,  26,  66,  99,  99,  99,  99,
     24,  26,  56,  99,  99,  99,  99,  99,
     47,  66,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99,
     99,  99,  99,  99,  99,  99,  99,  99], dtype=int)
std_chrominance_quant_tbl = std_chrominance_quant_tbl.reshape([8, 8])


def quant_tbl_quality_scale(quality):
    if (quality <= 0):
        quality = 1
    if (quality > 100):
        quality = 100
    if (quality < 50):
        qualityScale = 5000 / quality
    else:
        qualityScale = 200 - quality * 2
    luminanceQuantTbl = numpy.array(numpy.floor(
        (std_luminance_quant_tbl * qualityScale + 50) / 100))
    luminanceQuantTbl[luminanceQuantTbl == 0] = 1
    luminanceQuantTbl[luminanceQuantTbl > 255] = 255
    luminanceQuantTbl = luminanceQuantTbl.reshape([8, 8]).astype(int)

    chrominanceQuantTbl = numpy.array(numpy.floor(
        (std_chrominance_quant_tbl * qualityScale + 50) / 100))
    chrominanceQuantTbl[chrominanceQuantTbl == 0] = 1
    chrominanceQuantTbl[chrominanceQuantTbl > 255] = 255
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([8, 8]).astype(int)
    return luminanceQuantTbl, chrominanceQuantTbl

def rgb2ycbcr(im):
    cbcr = numpy.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    # Y
    cbcr[:,:,0] = 16 +  65.481/255 * r + 128.553/255 * g + 24.966/255 * b
    # Cb
    cbcr[:,:,1] = 128 - 37.797/255 * r - 74.203/255 * g + 112.0/255 * b
    # Cr
    cbcr[:,:,2] =  128 +  112.0/255 * r - 93.786/255 * g - 18.214/255 * b
    return numpy.uint8(cbcr)

def read_input_img(inputFile):
    image = Image.open(inputFile)
    withImg, heightImg = image.size
    imgMatrix = numpy.array(image)

    print('input image info:', image)
    print('size: \t', withImg, 'x', heightImg)
    return withImg, heightImg, imgMatrix

def main():
    # check the command in terminal
    if (len(sys.argv) != 4):
        print('___ERROR___')
        print('run code with the command: python encoder.py input_PPM_name.ppm output_JPEG_name.jpeg quality(from 1 to 100)')
        print('example:')
        print('python encoder.py ./input.ppm ./output.jpeg 95')
        return

    # defind some variable
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    quality = int(sys.argv[3])
    # set print option
    #numpy.set_printoptions(threshold=numpy.inf)

    # set up quantization table
    luminanceQuantTbl, chrominanceQuantTbl = quant_tbl_quality_scale(quality)

    # read input image and get with, high, data of .ppm image ogiginal
    withImg, heightImg, imgMatrix = read_input_img(inputFile)
    
    # add matrix [0, 0, 0...] if with, height of the image is not a multiple of 8
    withImg_ = withImg
    heightImg_ = heightImg

    if (withImg_ % 8 != 0):
        withImg_ = withImg_ // 8 * 8 + 8
    if (heightImg_ % 8 != 0):
        heightImg_ = heightImg_ // 8 * 8 + 8

    print('new size of the image(maybe similar to original size)',
          withImg_, 'x', heightImg_)

    # create a new matrix image and copy the value
    newImgMatrix = imgMatrix.copy()                

    # convert RGB to YCrCb
    newImgMatrix = rgb2ycbcr(newImgMatrix)

    # get channel Y, Cb, Cr
    Y_matrix = (newImgMatrix[:, :, 0] - 128).astype(numpy.int8)
    Cb_matrix = (newImgMatrix[:, :, 1] - 128).astype(numpy.int8)
    Cr_matrix = (newImgMatrix[:, :, 2] - 128).astype(numpy.int8)

    # divide block 8x8
    totalBlock = int((withImg_ / 8) * (heightImg_ / 8))
    currentBlock = 0

    Y_DC = numpy.zeros([totalBlock], dtype=int)
    Cb_DC = numpy.zeros([totalBlock], dtype=int)
    Cr_DC = numpy.zeros([totalBlock], dtype=int)
    d_Y_DC = numpy.zeros([totalBlock], dtype=int)
    d_Cb_DC = numpy.zeros([totalBlock], dtype=int)
    d_Cr_DC = numpy.zeros([totalBlock], dtype=int)

    sosBitStream = BitStream()
    for i in range(0, heightImg_, 8):
        for j in range(0, withImg_, 8):

            # use DCT transform form fft libary
            Y_DCTMatrix = fftpack.dct(fftpack.dct(Y_matrix[i:i + 8, j:j + 8], norm='ortho').T, norm='ortho').T
            Cb_DCTMatrix = fftpack.dct(fftpack.dct(Cb_matrix[i:i + 8, j:j + 8], norm='ortho').T, norm='ortho').T
            Cr_DCTMatrix = fftpack.dct(fftpack.dct(Cr_matrix[i:i + 8, j:j + 8], norm='ortho').T, norm='ortho').T
            
            #DCT tu tao
            #Y_DCTMatrix = my_DCT(Y_matrix[i:i + 8, j:j + 8])
            #Cb_DCTMatrix = my_DCT(Cb_matrix[i:i + 8, j:j + 8])
            #Cr_DCTMatrix = my_DCT(Cr_matrix[i:i + 8, j:j + 8])
            
            # quantization
            Y_QuantMatrix = numpy.rint(
                Y_DCTMatrix / luminanceQuantTbl).astype(int)
            Cb_QuantMatrix = numpy.rint(
                Cb_DCTMatrix / chrominanceQuantTbl).astype(int)
            Cr_QuantMatrix = numpy.rint(
                Cr_DCTMatrix / chrominanceQuantTbl).astype(int)

            # run length
            Y_ZZcode = Y_QuantMatrix.reshape([64])[zigzagOrder]
            Cb_ZZcode = Cb_QuantMatrix.reshape([64])[zigzagOrder]
            Cr_ZZcode = Cr_QuantMatrix.reshape([64])[zigzagOrder]
            
            #ham zigzac tu tao
            #zz = zigzac()
            #Y_ZZcode = Y_QuantMatrix.reshape([64])[zz]
            #Cb_ZZcode = Cb_QuantMatrix.reshape([64])[zz]
            #Cr_ZZcode = Cr_QuantMatrix.reshape([64])[zz]
            
            Y_DC[currentBlock] = Y_ZZcode[0]
            Cb_DC[currentBlock] = Cb_ZZcode[0]
            Cr_DC[currentBlock] = Cr_ZZcode[0]

            if (currentBlock == 0):
                d_Y_DC[currentBlock] = Y_DC[currentBlock]
                d_Cb_DC[currentBlock] = Cb_DC[currentBlock]
                d_Cr_DC[currentBlock] = Cr_DC[currentBlock]
            else:
                d_Y_DC[currentBlock] = Y_DC[currentBlock] - Y_DC[currentBlock-1]
                d_Cb_DC[currentBlock] = Cb_DC[currentBlock] - Cb_DC[currentBlock-1]
                d_Cr_DC[currentBlock] = Cr_DC[currentBlock] - Cr_DC[currentBlock-1]
            
            if (currentBlock == 0):
                print('std_luminance_quant_tbl:\n', std_luminance_quant_tbl)
                print('block8x8:\n', Y_matrix[i:i + 8, j:j + 8])
                print('DCT by fft:\n', Y_DCTMatrix)
                print('luminanceQuantTbl:\n', luminanceQuantTbl)
                print('quantizated block8x8:\n', Y_QuantMatrix)
                print('zigzag by zigzac function:\n', Y_ZZcode)
            # huffman encode https://www.impulseadventure.com/photo/jpeg-huffman-coding.html
            # encode y_DC
            sosBitStream.write(huffmanEncode.encodeDCToBoolList(
                d_Y_DC[currentBlock], 1, 1, currentBlock), bool)

            # encode y_AC

            huffmanEncode.encodeACBlock(
                sosBitStream, Y_ZZcode[1:], 1, 1, currentBlock)

            # encode Cb_DC

            sosBitStream.write(huffmanEncode.encodeDCToBoolList(
                d_Cb_DC[currentBlock], 0), bool)
            # encode Cb_AC

            huffmanEncode.encodeACBlock(
                sosBitStream, Cb_ZZcode[1:], 0)

            # encode Cr_DC

            sosBitStream.write(huffmanEncode.encodeDCToBoolList(
                d_Cr_DC[currentBlock], 0), bool)
            # encode Cr_AC

            huffmanEncode.encodeACBlock(
                sosBitStream, Cr_ZZcode[1:], 0)

            currentBlock = currentBlock + 1
           
          
    # create and open output file
    jpegFile = open(outputFile, 'wb+')

    # write jpeg header
    jpegFile.write(huffmanEncode.hexToBytes(
        'FFD8FFE000104A46494600010100000100010000'))

    # write Y Quantization Table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004300'))
    luminanceQuantTbl = luminanceQuantTbl.reshape([64])
    jpegFile.write(bytes(luminanceQuantTbl.tolist()))

    # write u/v Quantization Table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004301'))
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([64])
    jpegFile.write(bytes(chrominanceQuantTbl.tolist()))

    # write height and width
    jpegFile.write(huffmanEncode.hexToBytes('FFC0001108'))
    hHex = hex(heightImg)[2:]
    while len(hHex) != 4:
        hHex = '0' + hHex

    jpegFile.write(huffmanEncode.hexToBytes(hHex))

    wHex = hex(withImg)[2:]
    while len(wHex) != 4:
        wHex = '0' + wHex

    jpegFile.write(huffmanEncode.hexToBytes(wHex))

    # 03    01 11 00    02 11 01    03 11 01
    # 1：1	01 11 00	02 11 01	03 11 01
    # 1：2	01 21 00	02 11 01	03 11 01
    # 1：4	01 22 00	02 11 01	03 11 01

    # write Subsamp
    jpegFile.write(huffmanEncode.hexToBytes('03011100021101031101'))

    # write huffman table
    jpegFile.write(huffmanEncode.hexToBytes('FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))

    # SOS Start of Scan
    # yDC yAC uDC uAC vDC vAC
    sosLength = sosBitStream.__len__()
    filledNum = 8 - sosLength % 8
    if (filledNum != 0):
        sosBitStream.write(numpy.ones([filledNum]).tolist(), bool)

    # FF DA 00 0C 03 01 00 02 11 03 11 00 3F 00
    jpegFile.write(bytes([255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0]))

    # write encoded data
    sosBytes = sosBitStream.read(bytes)
    for i in range(len(sosBytes)):
        jpegFile.write(bytes([sosBytes[i]]))
        if (sosBytes[i] == 255):
            jpegFile.write(bytes([0]))  # FF to FF 00

    # write end symbol
    jpegFile.write(bytes([255, 217]))  # FF D9
    jpegFile.close()

    print("PSNR: ", psnr(inputFile, outputFile))
if __name__ == "__main__":
    # run code with the command: python encoder.py input_PPM_name.ppm output_JPEG_name.jpeg quality(from 1 to 100)
    main()
