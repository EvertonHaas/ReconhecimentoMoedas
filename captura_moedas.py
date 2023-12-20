import os
import cv2
from image_processing import pre_process
import numpy as np


def processar_moedas(diretorio_imagens, diretorio_destino):
    if not os.path.exists(diretorio_destino):
        os.makedirs(diretorio_destino)

    # Processar cada imagem no diretório
    for nome_arquivo in os.listdir(diretorio_imagens):
        if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            caminho_arquivo = os.path.join(diretorio_imagens, nome_arquivo)
            img = cv2.imread(caminho_arquivo)
            img = cv2.resize(img, (int(1840 / 4), int(3190 / 4)))
            img_pre = pre_process(img)

            countors, _ = cv2.findContours(img_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for i, cnt in enumerate(countors):
                area = cv2.contourArea(cnt)
                if area > 4000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    recorte = img[y:y + h, x:x + w]

                    recorte_redimensionado = cv2.resize(recorte, (224, 224))

                    # Salvar a imagem redimensionada
                    nome_arquivo_salvo = f"{os.path.splitext(nome_arquivo)[0]}_moeda_{i}.jpg"
                    cv2.imwrite(os.path.join(diretorio_destino, nome_arquivo_salvo), recorte_redimensionado)
