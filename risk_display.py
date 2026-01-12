

import cv2

def risk_kutusu_ekle(frame, risk):

    renkler = {
        'düşük': (0, 255, 0),      # Yeşil
        'orta': (0, 255, 255),     # Sarı
        'yüksek': (0, 0, 255)      # Kırmızı
    }

    renk = renkler.get(risk, (255, 255, 255))  # bilinmeyen riskte beyaz
    label = f'Risk: {risk.upper()}'

    x1, y1 = frame.shape[1] - 160, 10
    x2, y2 = frame.shape[1] - 10, 60

    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, -1)


    return frame

if __name__ == "__main__":
    import numpy as np

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for risk in ['düşük', 'orta', 'yüksek']:
        test_frame = risk_kutusu_ekle(dummy_frame.copy(), risk)
        cv2.imshow(f'Test - {risk}', test_frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
