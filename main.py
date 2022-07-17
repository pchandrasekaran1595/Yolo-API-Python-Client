import os
import re
import io
import sys
import cv2
import base64
import platform
import requests
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

BASE_PATH: str   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH: str  = os.path.join(BASE_PATH, 'input')
OUTPUT_PATH: str = os.path.join(BASE_PATH, 'output')

ID: int = 0
CAM_WIDTH: int  = 640
CAM_HEIGHT: int = 360 
FPS: int = 30


def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def decode_image(imageData) -> np.ndarray:
    _, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    return image


def encode_image_to_base64(header: str = "image/jpeg", image: np.ndarray = None) -> str:
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".jpeg", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'", "").replace("'", "")
    imageData = header + "," + imageData
    return imageData


def show_image(image: np.ndarray, cmap: str="gnuplot2", title: str=None) -> None:
    plt.figure()
    plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.axis("off")
    if title: plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


def main():
    args_1: tuple = ("--mode", "-m")
    args_2: tuple = ("--model", "-mo")
    args_3: tuple = ("--version", "-v")
    args_4: tuple = ("--filename", "-f")
    args_5: tuple = ("--downscale", "-ds")
    args_6: tuple = ("--url", "-u")

    mode: str = "image"
    model_type: str = "nano"
    version: int = 6
    filename: str = "Test_1.jpg"
    downscale: float = None
    base_url: str = "http://localhost:6600/infer"

    if args_1[0] in sys.argv: mode = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: mode = sys.argv[sys.argv.index(args_1[1]) + 1]
    
    if args_2[0] in sys.argv: model_type = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_2[1] in sys.argv: model_type = sys.argv[sys.argv.index(args_2[1]) + 1]

    if args_3[0] in sys.argv: version = int(sys.argv[sys.argv.index(args_3[0]) + 1])
    if args_3[1] in sys.argv: version = int(sys.argv[sys.argv.index(args_3[1]) + 1])

    if args_4[0] in sys.argv: filename = sys.argv[sys.argv.index(args_4[0]) + 1]
    if args_4[1] in sys.argv: filename = sys.argv[sys.argv.index(args_4[1]) + 1]

    if args_5[0] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_5[0]) + 1])
    if args_5[1] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_5[1]) + 1])

    if args_6[0] in sys.argv: base_url = sys.argv[sys.argv.index(args_6[0]) + 1]
    if args_6[1] in sys.argv: base_url = sys.argv[sys.argv.index(args_6[1]) + 1]

    assert re.match(r"^tiny$", model_type, re.IGNORECASE) or \
           re.match(r"^small$", model_type, re.IGNORECASE) or \
           re.match(r"^nano$", model_type, re.IGNORECASE), "Model type is not valid"

    url = f"{base_url}/v{version}/{model_type}"

    if re.match(r"^image$", mode, re.IGNORECASE):
        image = cv2.imread(os.path.join(INPUT_PATH, filename))
        payload = {
            "imageData": encode_image_to_base64(image=image),
        }
        response = requests.post(url=url, json=payload)

        if response.status_code == 200:
            if response.json()["statusCode"] == 200:
                cv2.rectangle(image, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
                show_image(image=image, title=f"{response.json()['label']} : {response.json()['score']}")
            else:
                breaker()
                print(response.json()["statusText"])
                breaker()
        else:
            print(f"Error {response.status_code} : {response.reason}")
        
    
    elif re.match(r"^video$", mode, re.IGNORECASE):
        cap = cv2.VideoCapture(os.path.join(INPUT_PATH, filename))

        while True:
            ret, frame = cap.read()
            if ret:
                if downscale:
                    frame = cv2.resize(
                        src=frame, 
                        dsize=(int(frame.shape[1]/downscale), int(frame.shape[0]/downscale)), 
                        interpolation=cv2.INTER_AREA
                    )
                frameData = encode_image_to_base64(image=frame)
                payload = {
                    "imageData" : frameData
                }       
                response = requests.post(base_url + "/" + model_type, json=payload)
                if response.status_code == 200:
                    if response.json()["statusCode"] == 200:
                        cv2.rectangle(frame, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
                        cv2.putText(frame, response.json()["label"], (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        print(response.json()["statusText"])
                        break
                else:
                    print(f"Error {response.status_code} : {response.reason}")
                    break

                cv2.imshow("Feed", frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        cv2.destroyAllWindows()

    
    elif re.match(r"^realtime$", mode, re.IGNORECASE):
        if platform.system() != "Windows":
            cap = cv2.VideoCapture(ID)
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frameData = encode_image_to_base64(image=frame)
            payload = {
                "imageData" : frameData
            }       
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                if response.json()["statusCode"] == 200:
                    cv2.rectangle(frame, (response.json()["box"][0], response.json()["box"][1]), (response.json()["box"][2], response.json()["box"][3]), (0, 255, 0), 2)
                    cv2.putText(frame, response.json()["label"], (response.json()["box"][0]-10, response.json()["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    print(response.json()["statusText"])
                    break
            else:
                print(f"Error {response.status_code} : {response.reason}")
                break
            
            cv2.imshow("Feed", frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
