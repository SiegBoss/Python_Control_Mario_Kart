# Control de Mario Kart con Visión Artificial | Control of Mario Kart with Artificial Vision

# Importar librerias | Import libraries
import cv2
import mediapipe as mp
import math
import keyboard


# Dibujar los puntos de referencia | Draw the reference points
drawing = mp.solutions.drawing_utils
# Analizar los puntos de referencia | Analyze the reference points
pose = mp.solutions.pose
# Detectar las manos, rostro y cuerpo | Detect hands, face and body
mp_holistic = mp.solutions.holistic
# Capturar la cámara web | Capture the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# -------------------

# Posición del círculo | Circle position
width = 700
height = 500

# Radio del Circulo | Circle Radius
circle_radius = 50

# Definir el límite de distancia | Define the distance limit
distancia_limite = 190

# -------------------


# Definir el centro y el radio del círculo | Define the center and radius of the circle
centro_circulo = (int(width * 0.75), int(height * 0.5))

# Iniciar la detección de pose | Start pose detection

# static_image_mode=False : Indica que se va a analizar un video en tiempo real y no una imagen estática | Indicates that a real-time video will be analyzed and not a static image
# model_complexity=0 : Indica que se va a utilizar el modelo más sencillo | Indicates that the simplest model will be used
with pose.Pose(static_image_mode=False, model_complexity=0) as body:
    
    # Iniciar el ciclo while para capturar los frames | Start the while loop to capture the frames
    while True:
        
        # Leer el frame | Read the frame
        # ret es un booleano que indica si se capturó el frame correctamente | ret is a boolean that indicates if the frame was captured correctly
        # frame es la imagen capturada | frame is the captured image
        ret, frame = cap.read()
        
        # Si no hay frame, se termina el ciclo | If there is no frame, the cycle ends
        if ret == False:
            break

        # Voltear el frame horizontalmente | Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        # Obtener el alto y ancho del frame | Get the height and width of the frame
        height, width, _ = frame.shape
        # Convertir el frame de BGR a RGB | Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesar el frame | Process the frame
        results = body.process(frame_rgb)

        # Analisa si se a detectado la pose | Analyze if the pose has been detected
        if results.pose_landmarks is not None:
            
            # Dibujar las marcas de pose | Draw the pose marks
            # results.pose_landmarks: Contiene las coordenadas de las marcas de pose | Contains the coordinates of the pose marks
            # mp_holistic.POSE_CONNECTIONS: Contiene las conexiones entre las marcas de pose | Contains the connections between the pose marks
            # drawing.DrawingSpec: Contiene los colores y el grosor de las marcas y conexiones | Contains the colors and thickness of the marks and connections
            # drawing._normalized_to_pixel_coordinates: Convierte las coordenadas normalizadas a pixeles | Converts normalized coordinates to pixels
            drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
                drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

            # Dibujar el círculo | Draw the circle
            cv2.circle(frame, centro_circulo, circle_radius, (0, 255, 0), 2)

            # Obtener las coordenadas de las manos | Get the coordinates of the hands
            right_hand = results.pose_landmarks.landmark[
                mp_holistic.PoseLandmark.RIGHT_INDEX
            ]
            left_hand = results.pose_landmarks.landmark[  
                mp_holistic.PoseLandmark.LEFT_INDEX
            ]

            # Convertir las coordenadas a pixeles | Convert coordinates to pixels
            right_hand_px = drawing._normalized_to_pixel_coordinates(
                right_hand.x, right_hand.y, width, height
            )
            left_hand_px = drawing._normalized_to_pixel_coordinates(
                left_hand.x, left_hand.y, width, height
            )

            # Si se detectan las dos manos | If both hands are detected
            if right_hand_px and left_hand_px:
                # Dibujar una línea entre las manos | Draw a line between the hands
                cv2.line(
                    frame, right_hand_px, left_hand_px, color=(255, 0, 0), thickness=2
                )

                # Calcula la diferencia entre las coordenadas de las manos | Calculate the difference between the coordinates of the hands
                dx = right_hand_px[0] - left_hand_px[0]
                dy = right_hand_px[1] - left_hand_px[1]
                # Calcular el ángulo entre las manos en Radianes | Calculate the angle between the hands in Radians
                angle = math.atan2(dy, dx)
                # Convertir el ángulo a grados | Convert the angle to degrees
                angle = math.degrees(angle)

                # Calcular la distancia entre las manos | Calculate the distance between the hands
                distancia = math.sqrt(dx**2 + dy**2)

                # Si la distancia es menor al límite | If the distance is less than the limit
                if distancia < distancia_limite:
                    
                    # Si el ángulo es mayor a 70 y menor a 165 | If the angle is greater than 70 and less than 165
                    if (angle > 70) and (angle < 165):
                        
                        # Presionar la tecla "a" | Press the "a" key
                        keyboard.press("a")
                        
                    else:
                        
                        keyboard.release("a")

                    # Si el ángulo es mayor a -165 y menor a -70 | If the angle is greater than -165 and less than -70
                    if (angle > -165) and (angle < -70):
                        
                        # Presionar la tecla "d" | Press the "d" key
                        keyboard.press("d")
                        
                    else:
                        
                        keyboard.release("d")

                # Si la distancia entre la mano y el centro del círculo | If the distance between the hand and the center of the circle
                dist = math.sqrt(
                    (left_hand_px[0] - centro_circulo[0]) ** 2
                    + (left_hand_px[1] - centro_circulo[1]) ** 2
                )

                # Si la distancia es menor al radio del círculo | If the distance is less than the radius of the circle
                if dist < circle_radius:
                    
                    keyboard.press("q")
                    
                else:
                    
                    keyboard.release("q")

        # Mostrar el frame | Show the frame
        cv2.imshow("Output", frame)

        # Si se presiona la tecla "esc" se termina el ciclo | If the "esc" key is pressed, the cycle ends
        if cv2.waitKey(1) & 0xFF == 27:
            
            break

# Liberar la cámara y destruir las ventanas | Release the camera and destroy the windows
cap.release()
cv2.destroyAllWindows()
