import numpy as np
import cv2 as cv
import os



litere = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
          'U', 'V', 'W', 'X', 'Y', 'Z']



def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    image = cv.resize(image, (800, 800))

    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def filter_blue(image):
    # Convert BGR to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # # Convert specific blue from RGB to HSV
    blue_rgb = np.uint8([[[2, 143, 178]]])  # RGB color
    blue_hsv = cv.cvtColor(blue_rgb, cv.COLOR_RGB2HSV)[0][0]

    # Define a small range around the converted HSV blue color
    lower_blue = np.array([blue_hsv[0] - 25, 50, 50])
    upper_blue = np.array([blue_hsv[0] + 10, 255, 255])


    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(image, image, mask=mask)

    return res
    
puncte_initiale = [(0,0),(0,0),(0,0),(0,0)]


def adjust_luminosity(image, target_brightness=85):
    # Calculate the mean brightness of the image
    mean_brightness = np.mean(image)

    # Calculate the adjustment factor
    adjustment_factor = target_brightness - mean_brightness

    # Adjust the brightness of the image
    adjusted_image = image + adjustment_factor

    # Clip the values to be in the range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    # show_image('adjusted_image',adjusted_image)

    return adjusted_image

def extrage_careu(image, path, first_image):
    # Filter colors in the image, only keeping blue and white; then convert to grayscale.
    # Bring image to a certain luminosity
    #  image = adjust_luminosity(image)

    original_ = image.copy()

    image = filter_blue(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image_m_blur = cv.medianBlur(image, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 1), 13) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.90 , image_g_blur, -0.90, 0)
    # show_image('image_sharpened',image_sharpened)
    _, thresh = cv.threshold(image_sharpened, 53, 255, cv.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=12)
 
    kernel = np.ones((3, 3), np.uint8)  # Increase the size for more dilation
    thresh = cv.dilate(thresh, kernel, iterations=20)  # Increase iterations for more dilation

    kernel = np.ones((1, 1), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=3)

    # Sharpen the image
    image_m_blur = cv.medianBlur(thresh, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 1), 17)
    thresh = cv.addWeighted(image_m_blur, 1.77 , image_g_blur, -1.1, 0)

    # show_image('thresh',thresh)

    md  = np.median(thresh)
    lower = int(max(0, (1.0 - 0.33) * md))
    upper = int(min(255, (1.0 + 0.33) * md))

    edges = cv.Canny(thresh, lower, upper)
    # show_image('edges',edges)
    contours, _ = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if 0.9 < cv.boundingRect(cnt)[2]/cv.boundingRect(cnt)[3] < 1]
    max_area = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 1500
    height = 1500

    # Padding
    top_left[0] -= 25
    top_left[1] -= 25

    top_right[0] += 25
    top_right[1] -= 25

    bottom_left[0] -= 20
    bottom_left[1] += 20

    bottom_right[0] += 20
    bottom_right[1] += 20

    # Draw the corners
    
    # image_copy = cv.cvtColor(image.copy(),cv.COLOR_GRAY2RGB)
    # cv.circle(image_copy,tuple(top_left),20,(0,0,255),-1)
    # cv.circle(image_copy,tuple(top_right),20,(0,0,255),-1)
    # cv.circle(image_copy,tuple(bottom_left),20,(0,0,255),-1)
    # cv.circle(image_copy,tuple(bottom_right),20,(0,0,255),-1)
    # # Draw the lines between the points
    # cv.line(image_copy, tuple(top_left), tuple(top_right), color=(0, 255, 0), thickness=5)
    # cv.line(image_copy, tuple(top_right), tuple(bottom_right), color=(0, 255, 0), thickness=5)
    # cv.line(image_copy, tuple(bottom_right), tuple(bottom_left), color=(0, 255, 0), thickness=5)
    # cv.line(image_copy, tuple(bottom_left), tuple(top_left), color=(0, 255, 0), thickness=5)
    # # show_image("detected corners",image_copy)
    
    puzzle_corners = np.array([[top_left],[top_right],[bottom_right],[bottom_left]], dtype=np.float32)
    if first_image == True:
        puncte_initiale[0] = top_left
        puncte_initiale[1] = top_right
        puncte_initiale[2] = bottom_right
        puncte_initiale[3] = bottom_left
    else: # if the puzzle corners are deviated more than 20 pixels compared to the first image, keep the first image's corners
        if abs(top_left[0] - puncte_initiale[0][0]) > 150 or abs(top_left[1] - puncte_initiale[0][1]) > 150:
            top_left = puncte_initiale[0]
        if abs(top_right[0] - puncte_initiale[1][0]) > 150 or abs(top_right[1] - puncte_initiale[1][1]) > 150:
            top_right = puncte_initiale[1]
        if abs(bottom_right[0] - puncte_initiale[2][0]) > 150 or abs(bottom_right[1] - puncte_initiale[2][1]) > 150:
            bottom_right = puncte_initiale[2]
        if abs(bottom_left[0] - puncte_initiale[3][0]) > 150 or abs(bottom_left[1] - puncte_initiale[3][1]) > 150:
            bottom_left = puncte_initiale[3]
        puzzle_corners = np.array([[top_left],[top_right],[bottom_right],[bottom_left]], dtype=np.float32)

    image_copy = cv.cvtColor(image.copy(),cv.COLOR_GRAY2RGB)
    cv.circle(image_copy,tuple(top_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(top_right),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_right),20,(0,0,255),-1)
    # Draw the lines between the points
    cv.line(image_copy, tuple(top_left), tuple(top_right), color=(0, 255, 0), thickness=5)
    cv.line(image_copy, tuple(top_right), tuple(bottom_right), color=(0, 255, 0), thickness=5)
    cv.line(image_copy, tuple(bottom_right), tuple(bottom_left), color=(0, 255, 0), thickness=5)
    cv.line(image_copy, tuple(bottom_left), tuple(top_left), color=(0, 255, 0), thickness=5)
    # show_image("detected corners",image_copy)
    
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
    perspective_transform = cv.getPerspectiveTransform(puzzle_corners, destination_of_puzzle)

    result = cv.warpPerspective(original_, perspective_transform, (width, height))

    hsv = cv.cvtColor(result, cv.COLOR_BGR2HSV)

    coffee_lower = np.array([46, 0, 0])
    coffee_upper = np.array([255, 255, 255])

    mask = cv.inRange(hsv, coffee_lower, coffee_upper)

    result = cv.bitwise_and(result, result, mask=mask)

    # show_image('result',result)

    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

    return result



def pattern_matching (patch):
    patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    patch = patch.astype('uint8')
    patch = cv.GaussianBlur(patch, (0, 0), 2)
    # Raise resolution
    patch = cv.resize(patch, (0, 0), fx=2, fy=2)
    # show_image('patch',patch)
    base_path_identifying_numbers = 'imagini_auxiliare\\'

    max_correlation = -np.inf
    poz = None

    for i in range(0, 7):
        folder = os.path.join(base_path_identifying_numbers, f'{i}')
        # show each image in the folder
        with os.scandir(folder) as entries:
            for entry in entries:
                image_path = os.path.join(folder, entry.name)
                if not os.path.exists(image_path):
                    print(f"Image path {image_path} does not exist.")
                image = cv.imread(image_path)
                if image is None:
                    print(f"Image at {image_path} could not be loaded.")
                # show_image('image',image)
                image_template = image.copy()
                image_template = cv.cvtColor(image_template, cv.COLOR_BGR2GRAY)
                image_template = cv.GaussianBlur(image_template, (0, 0), 2)
                image_template = cv.resize(image_template, (0, 0), fx=2, fy=2)
                # Crop the image to get only the center
                image_template = image_template[10:-10, 10:-10]
                # show_image('image_template',image_template)
                # Multiscale template matching
                for scale in np.linspace(0.1, 1.2, 25)[::-1]:
                    resized = cv.resize(image_template, (0, 0), fx=scale, fy=scale)
                    # show_image('resized',resized)
                    # if the resized image is smaller than the template, then break from the loop
                    # if resized.shape[0] < patch.shape[0] or resized.shape[1] < patch.shape[1]:
                    #     print
                    #     break
                    # Apply template Matching
                    result = cv.matchTemplate(patch, resized, cv.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv.minMaxLoc(result)
                    # check to see if the iteration should be visualized
                    # draw a bounding box around the detected region
                    if max_val > max_correlation:
                        max_correlation = max_val
                        poz = i
                        # print(poz)
    # print(poz)
    if poz != None:
        return poz

def determina_configuratie_cifre(original_image, image, lines_horizontal, lines_vertical, matrix, poz_i, poz_j): 
    offset = 30
    offset2 = 20
    matrix_numere = np.empty((15,15), dtype='int')
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = max(lines_vertical[j][0][0] - offset, 0)
            y_max = min(lines_vertical[j + 1][1][0] + offset2, 1500)
            x_min = max(lines_horizontal[i][0][1] - offset2, 0)
            x_max = min(lines_horizontal[i + 1][1][1] + offset, 1500)
            patch = image[x_min:x_max, y_min:y_max].copy()
            patch_original = original_image[x_min:x_max, y_min:y_max].copy()
            # patch_original = cv.cvtColor(patch_original, cv.COLOR_BGR2GRAY)
            patch_original = patch_original.astype('uint8')
            # print(y_min, y_max, x_min, x_max)
            if matrix[i][j] == 'x' and i == poz_i and j == poz_j:
                matrix_numere[i][j] = pattern_matching(patch_original)
                # cv.imshow('patch original', patch_original)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
            else:
                matrix_numere[i][j] = -1
            
            
    return matrix_numere

def stabilire_medie(image, lines_horizontal, lines_vertical):
    # Luam mediile celor 9 patratele din centru si le sortam
    offset = 20
    medii = []
    for i in range(6, 9):
        for j in range(6, 9):
            y_min = lines_vertical[j][0][0] + offset
            y_max = lines_vertical[j + 1][1][0] - offset
            x_min = lines_horizontal[i][0][1]  + offset
            x_max = lines_horizontal[i + 1][1][1] - offset
            patch = image[x_min:x_max, y_min:y_max].copy()
            medie_patch = np.mean(patch)
            medii.append(medie_patch)
    medii.sort()
    return medii




def determina_configuratie_careu_ox(image,lines_horizontal,lines_vertical, medie): 
    offset = 20
    matrix = np.empty((15,15), dtype='str')
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0] + offset
            y_max = lines_vertical[j + 1][1][0] - offset
            x_min = lines_horizontal[i][0][1]  + offset
            x_max = lines_horizontal[i + 1][1][1] - offset
            patch = image[x_min:x_max, y_min:y_max].copy()
            # cv.imshow('patch configurare careu', patch)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            medie_patch = np.mean(patch)
            if medie_patch < medie:
                matrix[i][j] = 'x'
            else:
                # print(medie_patch)
                matrix[i][j] = 'o'
            
            
    return matrix

def vizualizare_configuratie(result,matrix,lines_horizontal,lines_vertical):
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            if matrix[i][j] == 'x': 
                cv.rectangle(result, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=5)
                # put an X in the middle of the square
                cv.line(result, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=5)
                cv.line(result, (y_min, x_max), (y_max, x_min), color=(255, 0, 0), thickness=5)
    show_image('result',result)

def medie_totala(image, lines_horizontal, lines_vertical):
    # Calculez media tuturor patratelelor din care e alcatuita tabla
    # Sortez mediile si iau primele num_piese
    offset = 20
    medii = []
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0] + offset
            y_max = lines_vertical[j + 1][1][0] - offset
            x_min = lines_horizontal[i][0][1]  + offset
            x_max = lines_horizontal[i + 1][1][1] - offset
            patch = image[x_min:x_max, y_min:y_max].copy()
            medie_patch = np.mean(patch)
            medii.append(medie_patch)
    medii.sort()
    return medii


def comparare_matrici(img0, img):
    global litere, base_path

    image_path = os.path.join(base_path, img)
    if not os.path.exists(image_path):
            print(f"Image path {image_path} does not exist.")
    image = cv.imread(image_path)
    if image is None:
        print(f"Image at {image_path} could not be loaded.")
    if img0 == None:
        result=extrage_careu(image, image_path, True)
    else:
        result=extrage_careu(image, image_path, False)

    # show_image('result',result)

    off = 0

    lines_horizontal=[]
    for i in range(10,1520,97):
        l=[]
        l.append((0,i + off))
        l.append((1500 ,i + off))
        lines_horizontal.append(l)
        if len(lines_horizontal) == 8:
            off = 5

    off_vertical = 5
    lines_vertical=[]
    for i in range(23,1525,97):
        l=[]
        l.append((i,0))
        l.append((i,2000))
        lines_vertical.append(l)
        if len(lines_vertical) == 8:
            off_vertical = 0

    # for line in  lines_vertical : 
    #     cv.line(result, line[0], line[1], (0, 255, 0), 5)
    #     for line in  lines_horizontal : 
    #         cv.line(result, line[0], line[1], (0, 0, 255), 5)
    # show_image('result',result)
    _, thresh = cv.threshold(result, 190, 230, cv.THRESH_BINARY_INV)
    if img0 is None: # e prima piesa
        medii = stabilire_medie(thresh, lines_horizontal, lines_vertical)
        # show_image('thresh',thresh)
        matrix = determina_configuratie_careu_ox(thresh,lines_horizontal,lines_vertical, medii[2])
    else:
        medii = medie_totala(thresh, lines_horizontal, lines_vertical)
        matrix = determina_configuratie_careu_ox(thresh,lines_horizontal,lines_vertical, medii[int(img[2 : 4]) * 2])

    #vizualizare_configuratie(result.copy(),matrix,lines_horizontal,lines_vertical)

    if img0 is None: # e prima piesa
        # Empty the file
        with open(f'Folder_Solutie\\{img[:-4]}.txt', 'w') as f:
            f.write('')
        for i in range(15):
            for j in range(15):
                if matrix[i][j] != 'o':
                    print(f"Found a difference at {i + 1}{litere[j]}", end=' ')
                    matrix_numbers = determina_configuratie_cifre(result, thresh, lines_horizontal, lines_vertical, matrix, i, j)
                    print(str(matrix_numbers[i][j]))
                    # Make a file in "Folder_Solutie" with the image name 
                    with open(f'Folder_Solutie\\{img[:-4]}.txt', 'a') as f:
                        # write the letters
                        f.write(f"{i + 1}{litere[j]} {str(matrix_numbers[i][j])}")
                        f.write('\n')
                        f.close()
        return 

    # ---------------------------------------------

    image_path0 = os.path.join(base_path, img0)
    if not os.path.exists(image_path0):
            print(f"Image path {image_path0} does not exist.")
    image0 = cv.imread(image_path0)
    if image0 is None:
        print(f"Image at {image_path0} could not be loaded.")

    result0=extrage_careu(image0, image_path0, False)


    off = 0

    lines_horizontal0=[]
    for i in range(10,1520,97):
        l=[]
        l.append((0,i + off))
        l.append((1500 ,i + off))
        lines_horizontal0.append(l)
        if len(lines_horizontal0) == 8:
            off = 5

    off_vertical = 5
    lines_vertical0=[]
    for i in range(23,1525,97):
        l=[]
        l.append((i + off_vertical,0))
        l.append((i + off_vertical,2000))
        lines_vertical0.append(l)
        if len(lines_vertical0) == 8:
            off_vertical = 0

    # for line in  lines_vertical0 :
    #     cv.line(result0, line[0], line[1], (0, 255, 0), 5)
    #     for line in  lines_horizontal0 :
    #         cv.line(result0, line[0], line[1], (0, 0, 255), 5)

    medii0 = medie_totala(result0, lines_horizontal0, lines_vertical0)
    _, thresh0 = cv.threshold(result0, 190, 230, cv.THRESH_BINARY_INV)
    medii0 = medie_totala(thresh0, lines_horizontal0, lines_vertical0)
    matrix0 = determina_configuratie_careu_ox(thresh0,lines_horizontal0,lines_vertical0, medii0[int(img0[2 : 4]) * 2] )
    #vizualizare_configuratie(result0.copy(),matrix0,lines_horizontal0,lines_vertical0)

    # ---------------------------------------------

    # Empty the file
    with open(f'Folder_Solutie\\{img0[:-4]}.txt', 'w') as f:
        f.write('')
    piesa = []
    pozitii = []
    nr_diff = 0
    for i in range(15):
        for j in range(15):
            if matrix[i][j] != matrix0[i][j]:
                nr_diff += 1
                print(f"Found a difference at {i + 1}{litere[j]}", end=' ')
                matrix_numbers = determina_configuratie_cifre(result0, thresh0, lines_horizontal0, lines_vertical0, matrix0, i, j)
                print(str(matrix_numbers[i][j]))
                piesa.append(matrix_numbers[i][j])
                pozitii.append((i, j))
                # Make a file in "Folder_Solutie" with the image name
                with open(f'Folder_Solutie\\{img0[:-4]}.txt', 'a') as f:
                    # Append the letters
                    f.write(f"{i + 1}{litere[j]} {str(matrix_numbers[i][j])}")
                    f.write('\n')
                    f.close()
    if nr_diff != 2:
        # Exit the program
        print('ERROR ERROR ERROR')
        vizualizare_configuratie(result.copy(),matrix,lines_horizontal,lines_vertical)
        vizualizare_configuratie(result0.copy(),matrix0,lines_horizontal0,lines_vertical0)
        #exit()
    return piesa, pozitii

    

def main(base_path, traseu, matrice_scor):
    for i in range(1, 6): #modify
        scor_jucator_1 = -1
        scor_jucator_2 = -1
        with open(f'{base_path}\\{i}_mutari.txt', 'r') as f:
            mutare = f.read()
            mutare = mutare.replace(' ', '-')
            mutare = mutare.replace('\n', ' ')
            mutare = mutare.split(' ')
        for j in range(1, 21): #modify
            if j < 10:
                img0 = f'{i}_0{j}.jpg'
            else:
                img0 = f'{i}_{j}.jpg'
            # show_image('result0',result0)
            if j == 1: # e prima piesa
                print(img0)
                comparare_matrici(None, img0)
                with open(f'Folder_Solutie\\{img0[:-4]}.txt', 'a') as f:
                    f.write('0')
                    f.close()
            else:
                if j - 1 < 10:
                    img = f'{i}_0{j - 1}.jpg'
                else:
                    img = f'{i}_{j - 1}.jpg'
                print(img0 + ' <> ', end='')
                print(img)
                piesa, pozitii = comparare_matrici(img0, img)
                
                # Verificam ce player a facut mutarea din fisierul i_mutari.txt
                scor_runda = 0
                # Transform everyting to int
                piesa[0] = int(piesa[0])
                piesa[1] = int(piesa[1])
                pozitii[0] = (int(pozitii[0][0]), int(pozitii[0][1]))
                pozitii[1] = (int(pozitii[1][0]), int(pozitii[1][1]))
                if int(mutare[j - 1][-1]) == 1:
                    # Verificam daca noua piesa pusa are unul dintre numerele care se afla sub
                    # unul dintre pioni
                    if scor_jucator_1 != -1:
                        if piesa[0] == traseu[scor_jucator_1] or piesa[1] == traseu[scor_jucator_1]:
                            scor_jucator_1 += 3
                            scor_runda += 3
                    if scor_jucator_2 != -1:
                        if piesa[0] == traseu[scor_jucator_2] or piesa[1] == traseu[scor_jucator_2]:
                            scor_jucator_2 += 3
                    if matrice_scor[pozitii[0][0]][pozitii[0][1]] != 0:
                        scor_jucator_1 += matrice_scor[pozitii[0][0]][pozitii[0][1]]
                        scor_runda += matrice_scor[pozitii[0][0]][pozitii[0][1]]
                        if piesa[0] == piesa[1]:
                            scor_jucator_1 += matrice_scor[pozitii[0][0]][pozitii[0][1]]
                            scor_runda += matrice_scor[pozitii[0][0]][pozitii[0][1]]
                    if matrice_scor[pozitii[1][0]][pozitii[1][1]] != 0:
                        scor_jucator_1 += matrice_scor[pozitii[1][0]][pozitii[1][1]]
                        scor_runda += matrice_scor[pozitii[1][0]][pozitii[1][1]]
                        if piesa[0] == piesa[1]:
                            scor_jucator_1 += matrice_scor[pozitii[1][0]][pozitii[1][1]]
                            scor_runda += matrice_scor[pozitii[1][0]][pozitii[1][1]]
                else:
                    if scor_jucator_1 != -1:
                        if piesa[0] == traseu[scor_jucator_1] or piesa[1] == traseu[scor_jucator_1]:
                            scor_jucator_1 += 3
                    if scor_jucator_2 != -1:
                        if piesa[0] == traseu[scor_jucator_2] or piesa[1] == traseu[scor_jucator_2]:
                            scor_jucator_2 += 3
                            scor_runda += 3
                    if matrice_scor[pozitii[0][0]][pozitii[0][1]] != 0:
                        scor_jucator_2 += matrice_scor[pozitii[0][0]][pozitii[0][1]]
                        scor_runda += matrice_scor[pozitii[0][0]][pozitii[0][1]]
                        if piesa[0] == piesa[1]:
                            scor_jucator_2 += matrice_scor[pozitii[0][0]][pozitii[0][1]]
                            scor_runda += matrice_scor[pozitii[0][0]][pozitii[0][1]]
                    if matrice_scor[pozitii[1][0]][pozitii[1][1]] != 0:
                        scor_jucator_2 += matrice_scor[pozitii[1][0]][pozitii[1][1]]
                        scor_runda += matrice_scor[pozitii[1][0]][pozitii[1][1]]
                        if piesa[0] == piesa[1]:
                            scor_jucator_2 += matrice_scor[pozitii[1][0]][pozitii[1][1]]
                            scor_runda += matrice_scor[pozitii[1][0]][pozitii[1][1]]
                print(f"Scorul jucatorului 1 este {scor_jucator_1}")
                print(f"Scorul jucatorului 2 este {scor_jucator_2}")
                print(f"Scorul rundei este {scor_runda}")
                with open(f'Folder_Solutie\\{img0[:-4]}.txt', 'a') as f:
                    f.write(str(scor_runda))
                    f.close()



            
        print('-------------------')

    



base_path = 'testare\\'

traseu = [1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0, 3, 5, 4, 1, 6, 2, 4, 5, 5, 0, 6, 3, 4, 2, 0, 1, 5, 1, 3, 4, 4, 4, 5, 0, 6, 3, 5, 4, 1, 3, 2, 0, 0, 1, 1, 
          2, 3, 6, 3, 5, 2, 1, 0, 6, 6, 5, 2, 1, 2, 5, 0, 6, 6, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4, 2, 6, 2, 3, 1, 6, 5, 6, 2, 0, 4, 0, 1, 6, 4, 4, 1, 6, 6, 3, 0]

matrice_scor = [
    [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
    [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
    [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
    [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
    [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
    [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
    [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
    [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
    [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
    [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
    [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
]

# # --------------------------------------------- MAIN --------------------------------------------- # #
main(base_path, traseu, matrice_scor)








# # --------------------------------------------- DEBUG --------------------------------------------- # #

# for i in range(1, 6):
#     for j in range(1, 21):
#         if j < 10:
#             img0 = f'{i}_0{j}.jpg'
#         else:
#             img0 = f'{i}_{j}.jpg'
#         # show_image('result0',result0)
#         image_path0 = os.path.join(base_path, img0)
#         if not os.path.exists(image_path0):
#                 print(f"Image path {image_path0} does not exist.")
#         image0 = cv.imread(image_path0)
#         if image0 is None:
#             print(f"Image at {image_path0} could not be loaded.")
#         if j == 1: # e prima piesa
#             result0=extrage_careu(image0, image_path0, True)
#         else:
#             result0=extrage_careu(image0, image_path0, False)
#         show_image('result0',result0)

# img = '1_04.jpg'

# image_path = os.path.join(base_path, img)
# if not os.path.exists(image_path):
#         print(f"Image path {image_path} does not exist.")
# image = cv.imread(image_path)
# if image is None:
#     print(f"Image at {image_path} could not be loaded.")

# result=extrage_careu(image, image_path, True)
# show_image('result',result)

# off = 0

# lines_horizontal=[]
# for i in range(10,1520,97):
#     l=[]
#     l.append((0,i + off))
#     l.append((1500 ,i + off))
#     lines_horizontal.append(l)
#     if len(lines_horizontal) == 8:
#         off = 5

# off_vertical = 5
# lines_vertical=[]
# for i in range(23,1525,97):
#     l=[]
#     l.append((i + off_vertical,0))
#     l.append((i + off_vertical,2000 ))
#     lines_vertical.append(l)
#     if len(lines_vertical) == 8:
#         off_vertical = 0

# result_copy = result.copy()
# for line in  lines_vertical : 
#     cv.line(result_copy, line[0], line[1], (0, 255, 0), 5)
#     for line in  lines_horizontal : 
#         cv.line(result_copy, line[0], line[1], (0, 0, 255), 5)

# # show_image('result',result_copy)
# _, thresh = cv.threshold(result, 190, 230, cv.THRESH_BINARY_INV)

# medii = medie_totala(thresh, lines_horizontal, lines_vertical)
# matrix = determina_configuratie_careu_ox(thresh,lines_horizontal,lines_vertical, medii[int(img[2 : 4]) * 2])
# # vizualizare_configuratie(result.copy(),matrix,lines_horizontal,lines_vertical)
# # print(matrix)
# nr_diff = 0
# for i in range(15):
#     for j in range(15):
#         if matrix[i][j] != matrix[i][j]:
#             nr_diff += 1
#             print(f"Found a difference at {i + 1}{litere[j]}", end=' ')
#             matrix_numbers = determina_configuratie_cifre(result, thresh, lines_horizontal, lines_vertical, matrix, i, j)
#             print(str(matrix_numbers[i][j]))
# print("-------------------")
# print(matrix_numbers)
# # Write the numbers on the image
# for i in range(len(lines_horizontal) - 1):
#     for j in range(len(lines_vertical) - 1):
#         y_min = lines_vertical[j][0][0]
#         y_max = lines_vertical[j + 1][1][0]
#         x_min = lines_horizontal[i][0][1]
#         x_max = lines_horizontal[i + 1][1][1]
#         if matrix_numbers[i][j] != -1:
#             cv.putText(result, str(matrix_numbers[i][j]), (y_min + 20, x_min + 60), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
#             cv.rectangle(result, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=5)
#             # write the letter
#             cv.putText(result, matrix_numbers[i][j], (y_min + 20, x_min + 120), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

# show_image('result',result)








# detecteaza_linii_orizontale(result)




# DE PUS README!!!