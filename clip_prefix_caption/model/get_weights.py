# import requests

# def download_file_from_google_drive(id, destination):
#     URL = "https://docs.google.com/uc?export=download"

#     session = requests.Session()

#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = get_confirm_token(response)

#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)

#     save_response_content(response, destination)    

# def get_confirm_token(response):
#     for key, value in response.cookies.items():
#         if key.startswith('download_warning'):
#             return value

#     return None

# def save_response_content(response, destination):
#     CHUNK_SIZE = 32768

#     with open(destination, "wb") as f:
#         for chunk in response.iter_content(CHUNK_SIZE):
#             if chunk: # filter out keep-alive new chunks
#                 f.write(chunk)

# if __name__ == "__main__":
#     file_id = '14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT'
#     destination = './model_wieghts.pt'
#     download_file_from_google_drive(file_id, destination)

# from google_drive_downloader import GoogleDriveDownloader as gdd

# gdd.download_file_from_google_drive(file_id='14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT',
#                                     dest_path='./model_wieghts.pt')

# import requests

# def download_file_from_google_drive(id, destination):
#     def get_confirm_token(response):
#         for key, value in response.cookies.items():
#             if key.startswith('download_warning'):
#                 return value

#         return None

#     def save_response_content(response, destination):
#         CHUNK_SIZE = 32768

#         with open(destination, "wb") as f:
#             for chunk in response.iter_content(CHUNK_SIZE):
#                 if chunk: # filter out keep-alive new chunks
#                     f.write(chunk)

#     URL = "https://docs.google.com/uc?export=download"

#     session = requests.Session()

#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = get_confirm_token(response)

#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)

#     save_response_content(response, destination)    


# if __name__ == "__main__":
#     # import sys
#     # if len(sys.argv) is not 3:
#     #     print("Usage: python google_drive.py drive_file_id destination_file_path")
#     # else:
#     # TAKE ID FROM SHAREABLE LINK
#     file_id = "14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT"
#     # DESTINATION FILE ON YOUR DISK
#     destination = './model_wieghts.pt'
#     download_file_from_google_drive(file_id, destination)

import wget

wget.download('https://drive.google.com/u/0/uc?id=1GYPToCqFREwi285wPLhuVExlz7DDUDfJ&export=download&confirm=t&uuid=e5844af7-98d6-48ee-82d8-1e6037c88d73')