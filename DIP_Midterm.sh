#!/bin/bash
# Linux/Mac build script for executable
# Linux/Mac için executable oluşturma scripti

echo "========================================"
echo "Image Processing App - EXE Builder"
echo "Görüntü İşleme Uygulaması - EXE Oluşturucu"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "HATA: Python bulunamadı!"
    echo "Lütfen Python'u yükleyin."
    exit 1
fi

echo "Python bulundu."
echo ""

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller bulunamadı. Yükleniyor..."
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo "HATA: PyInstaller yüklenemedi!"
        exit 1
    fi
fi

echo "PyInstaller hazır."
echo ""

# Install requirements
echo "Bağımlılıklar kontrol ediliyor..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "UYARI: Bazı bağımlılıklar yüklenemedi!"
fi

echo ""
echo "========================================"
echo "Executable oluşturuluyor..."
echo "========================================"
echo ""

# Build executable
pyinstaller build_exe.spec --clean

if [ $? -ne 0 ]; then
    echo ""
    echo "HATA: Executable oluşturulamadı!"
    exit 1
fi

echo ""
echo "========================================"
echo "Başarılı!"
echo "========================================"
echo ""
echo "Executable dosyası: dist/ImageProcessingApplication"
echo ""
echo "Test etmek için dist klasörüne gidin ve"
echo "ImageProcessingApplication dosyasını çalıştırın."
echo ""
