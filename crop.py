import cv2
import numpy as np
import os
from scipy.cluster.hierarchy import fcluster, linkage

def desenhar_contornos(imagem, linhas):
    # Obter as dimensões da imagem original
    altura, largura = imagem.shape[:2]
    
    nova_imagem = np.ones((altura, largura, 3), dtype=np.uint8) *255
    
    for linha in linhas:
        for contorno in linha:
            x,y,w,h = contorno

            nova_imagem[y:y+h, x:x+w] = imagem[y:y+h,x:x+w]
            cv2.imshow('Contornos', nova_imagem)
            cv2.waitKey(0)
    # Mostrar a nova imagem com os contornos
    cv2.destroyAllWindows()
    return nova_imagem

def groupby_contours(contours):
    used_contours = set()
    output = []
    max_width = max([ cv2.boundingRect(contour)[2] for contour in contours])
    for i in range(0,len(contours)):
        if i in used_contours:
            continue
        contour_aux = contours[i]
        xa,ya,wa,ha = cv2.boundingRect(contour_aux)
        for j in range(i+1,len(contours)):
            if j in used_contours:
                continue
            compared_contour = contours[j]
            xc,yc,wc,hc = cv2.boundingRect(compared_contour)
            if xa - xc < 5 and  (ya - yc < 15 and yc + hc < ya + ha):
                merged_x = min(xa, xc)
                merged_y = min(ya, yc)   
                merged_w = max(xa + wa, xc + wc) - merged_x   
                merged_h = max(ya + ha, yc + hc) - merged_y
                if merged_w > max_width:
                    break
                output.append(np.array([merged_x,merged_y,merged_w,merged_h]))
                used_contours.add(i)
                used_contours.add(j)
                break 

        if i not in used_contours and xa + wa > 15 and ya + ha > 15 : #made for filter semicolons and dots
            output.append(np.array([xa,ya,wa,ha]))
    return output

def sort_contours(contours):
    mean_y = [y for _,y,_,_ in contours]
    mean_H = [h for _,_,_,h in contours]
    def sorting_key(cnt):
        x, y, w, h = cnt
        return (y+h, x)
    return sorted(contours, key=sorting_key)

def cluster_lines(contours, vertical_threshold=20):
    # Calcular o centro vertical de cada bounding box
    centers = np.array([y + h / 2 for _, y, _, h in contours])

    # Aplicar clustering hierárquico com base na distância vertical
    Z = linkage(centers.reshape(-1, 1), method='single')

    # Agrupar os contornos que estão próximos verticalmente (mesma linha)
    clusters = fcluster(Z, t=vertical_threshold, criterion='distance')

    # Agrupar contornos em linhas
    lines = []
    for cluster_id in np.unique(clusters):
        line_contours = [cnt for i, cnt in enumerate(contours) if clusters[i] == cluster_id]
        line_contours_sorted = sorted(line_contours, key=lambda cnt: cnt[0])  # Ordenar por x
        lines.append(line_contours_sorted)

    # Ordenar as linhas pelo valor y mínimo do primeiro contorno em cada linha
    lines_sorted_by_y = sorted(lines, key=lambda line: min(cnt[1] for cnt in line))

    return lines_sorted_by_y

def crop_characters(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = groupby_contours(contours)
            linhas = cluster_lines(contours)
            
            text = os.path.splitext(filename)[0]
            i = 0
            for linha in linhas:                
                for contour in linha:
                    x, y, w, h = contour
                    padding = 2
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img.shape[1] - x, w + 2*padding)
                    h = min(img.shape[0] - y, h + 2*padding)
                    
                    char_image = img[y:y+h, x:x+w]
                    output_filename = f"{os.path.splitext(filename)[0]}_{i:03d}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, char_image)
                    i+=1


