import cv2
from image_processing import pre_process, detectar_moeda
from config import image_path
import captura_moedas
import os
import shutil
from treina_modelo import treinar_modelo
# tf 2.9.1
# keras 2.6.0


def main():
    img = cv2.imread(image_path)
    img = cv2.resize(img, (int(1840 / 4), int(3190 / 4)))
    img_pre = pre_process(img)

    qtd = 0
    countors, hi = cv2.findContours(img_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 4000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            recorte = img[y:y + h, x:x + w]
            classe, conf = detectar_moeda(recorte)
            if conf > 0.7:
                cv2.putText(img, str(classe), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if classe == '1 real':
                    qtd += 1
                if classe == '25 centavos':
                    qtd += 0.25
                if classe == '50 centavos':
                    qtd += 0.5
                if classe == '5 centavos':
                    qtd += 0.05
                if classe == '10 centavos':
                    qtd += 0.1

    cv2.rectangle(img, (430, 30), (600, 80), (0, 0, 255), -1)
    cv2.putText(img, f'R$ {qtd}', (440, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow('IMG', img)
    cv2.imshow('IMG PRE', img_pre)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def deletar_arquivos_diretorio(diretorio):
    if os.path.exists(diretorio):
        # Usando shutil.rmtree para deletar o diretório e recriá-lo é uma maneira rápida de limpar um diretório
        shutil.rmtree(diretorio)
        os.makedirs(diretorio)


if __name__ == "__main__":
    print("Escolha uma opção:")
    print("1 - Classificar Moedas")
    print("2 - Capturar Imagens Moedas")
    print("3 - Treinar Modelo")
    print("4 - Sair")
    opcao = int(input("Escolha uma opção: "))
    if opcao == 1:
        main()
    elif opcao == 2:
        diretorio_imagens = 'fotos/1 real/'
        diretorio_destino = 'fotos/data/treino/1 real/'
        deletar_arquivos_diretorio(diretorio_destino)
        captura_moedas.processar_moedas(diretorio_imagens, diretorio_destino)
        diretorio_imagens = 'fotos/50 centavos/'
        diretorio_destino = 'fotos/data/treino/50 centavos/'
        deletar_arquivos_diretorio(diretorio_destino)
        captura_moedas.processar_moedas(diretorio_imagens, diretorio_destino)
        diretorio_imagens = 'fotos/25 centavos/'
        diretorio_destino = 'fotos/data/treino/25 centavos/'
        deletar_arquivos_diretorio(diretorio_destino)
        captura_moedas.processar_moedas(diretorio_imagens, diretorio_destino)
        diretorio_imagens = 'fotos/10 centavos/'
        diretorio_destino = 'fotos/data/treino/10 centavos/'
        deletar_arquivos_diretorio(diretorio_destino)
        captura_moedas.processar_moedas(diretorio_imagens, diretorio_destino)
        diretorio_imagens = 'fotos/5 centavos/'
        diretorio_destino = 'fotos/data/treino/5 centavos/'
        deletar_arquivos_diretorio(diretorio_destino)
        captura_moedas.processar_moedas(diretorio_imagens, diretorio_destino)
        diretorio_imagens = 'fotos/Experimento/'
        diretorio_destino = 'fotos/data/treino/Experimento/'
        deletar_arquivos_diretorio(diretorio_destino)
        captura_moedas.processar_moedas(diretorio_imagens, diretorio_destino)
    elif opcao == 3:
        treinar_modelo()
    elif opcao == 4:
        exit()
    else:
        print("Opção inválida!")
