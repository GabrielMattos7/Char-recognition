import cv2
import numpy as np
import os

def groupby_contours(contours):
    used_contours = set()  # To track processed contours
    output = []
    for i in range(0,len(contours)):
        contour_aux = contours[i]
        xa,ya,wa,ha = cv2.boundingRect(contour_aux) #// a in the end as `a`uxiliar
        for j in range(i+1,len(contours)):
            if j in used_contours:
                continue
            compared_contour = contours[j]
            xc,yc,wc,hc = cv2.boundingRect(compared_contour)
            # if xa <= xc and xa + wa >= xc + wc and ya - yc > 0:
            if xa - xc < 5 and  (ya - yc < 10 and yc + hc < ya + ha):
                print(f'{i} and {j} must be complements')
                merged_x = min(xa, xc)
                merged_y = min(ya, yc)   
                merged_w = max(xa + wa, xc + wc) - merged_x   
                merged_h = max(ya + ha, yc + hc) - merged_y
                output.append(np.array([merged_x,merged_y,merged_w,merged_h]))
                used_contours.add(i)
                used_contours.add(j)
                break # lets supose that this can happend only once
        if i not in used_contours:
            output.append(np.array([xa,ya,wa,ha]))
    return output

def sort_contours(contours):
    def sorting_key(cnt):
        x, y, w, h = cnt
        return (y+h, x)  # Group by y bucket, then sort by x
    return sorted(contours, key=sorting_key)


def crop_characters(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours from left to right, then top to bottom
            contours = groupby_contours(contours)
            contours = sort_contours(contours)
            
            # Get the corresponding text
            text = os.path.splitext(filename)[0]
            
            for i, contour in enumerate(contours):
                x, y, w, h = contour
                # 
                # Add some padding
                padding = 2
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2*padding)
                h = min(img.shape[0] - y, h + 2*padding)
                
                char_image = img[y:y+h, x:x+w]
                output_filename = f"{os.path.splitext(filename)[0]}_{i:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, char_image)
                # print(f"Saved {output_filename}")

if __name__ == "__main__":
    input_directory = "./output_images"
    output_directory = "./cropped_characters"
    crop_characters(input_directory, output_directory)
