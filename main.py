import os
import uvicorn
import traceback
import tensorflow as tf
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, Response
from models import place  

app = FastAPI()

# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "Finally end point its done!"

# If your model need text input use this endpoint!
    text:str

@app.post("/recomendations")
def predict_text(req: place, response: Response):
    try:
        # In here you will get text sent by the user
        place = req.place
        print("Uploaded text:", place)

        data = pd.read_csv('tourism_with_id.csv')
        data.head()
        
        # Step 1: (Optional) Do your text preprocessing
        place_tourism = data.drop(['Description', 'City', 'Price', 'Rating', 'Time_Minutes', 'Coordinate', 'Lat', 'Long', 'Unnamed: 11', 'Unnamed: 12'], axis=1)

        place_tourism.head()

        """MISSING DATA"""

        place_tourism.isnull().sum()

        # Step 2: Prepare your data to your model
        tf = TfidfVectorizer()

        tf.fit(place_tourism['Category'])

        tf.get_feature_names_out()

        matriks_tfidf = tf.fit_transform(place_tourism['Category'])
        matriks_tfidf.toarray() #Masih berbentu matriks sparse

        print(f'Dimensi Matriks TFIDF : {matriks_tfidf.shape}')

        #Mengubah matriks sparse jadi matriks dense
        matriks_tfidf.todense()

        #Menggabungkan Data yang telah di Vektorisasi
        data_matriks = pd.DataFrame(
        matriks_tfidf.todense(),
        columns = tf.get_feature_names_out(),
        index = place_tourism.Place_Name
        )
        data_matriks

        """COSINE SIMILARITY"""

        #Buat nyari derajat persamaan tiap variabel nama wisata
        from sklearn.metrics.pairwise import cosine_similarity

        cos_sim = cosine_similarity(data_matriks)
        cos_sim

        data_cos_sim = pd.DataFrame(
            cos_sim,
            columns = place_tourism.Place_Name,
            index = place_tourism.Place_Name
        )
        data_cos_sim.sample(437,axis=0)

        # Step 3: Predict the data
        def recomendations(place_name, similarity_data = data_cos_sim, items = place_tourism[['Place_Name', 'Category']], k = 10):
            index = similarity_data.loc[:, place_name].to_numpy().argpartition(range(-1, -k, -1))
            closest =similarity_data.columns[index[-1:-(k+2):-1]]
            closest = closest.drop(place_name, errors = 'ignore')
            return pd.DataFrame(closest).merge(items).head(k)
    
        # result = model.predict
        # Predict the data
        rec_list = []
        rec = recomendations(place, similarity_data=data_cos_sim, items=place_tourism[['Place_Name', 'Category']], k=10)

        for user in rec.iterrows():
            rec_dict = {
                    user[1]['Place_Name']
            }
            rec_list.append(rec_dict)

            if len(rec_list) >= 10:
                break

        return {
    'data': rec_list
}

        return {
            'data': rec_list
        }
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"
    
        # Step 4: Change the result your determined API output
        
        return "Endpoint not implemented"
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

# Starting the server
# Your can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)