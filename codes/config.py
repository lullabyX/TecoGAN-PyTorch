import os

# save weight directory
save_weight_dir = ''
while True: 
    print('Enter directory for to save model weights: ')
    save_weight_dir = input()
    if not os.path.exists(save_weight_dir):
        print('Directory Not Found')
        continue;
    break

# save image directory
save_image_dir = 'drive/MyDrive/Thesis/Train/'

while True: 
    print('Enter drive directory to save image:')
    save_image_dir = input()
    if os.path.exists(save_image_dir):
        print(save_image_dir)
        break
    print('Directory not found')


# Check if center crop
print('Do you want to center crop image?(yes/no): ')
center_crop= ''
while True:
    center_crop = input()
    if center_crop == 'yes' or center_crop == 'no':
        break