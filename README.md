# "Updating LSTM"
### Kort om projektet
Projektet heter så här för att modellen, som är en LSTM, får under inferens sina vikter uppdaterade
när en ny candlestick släpps. När en prediktion görs (manuellt via menyn) ställs modellens *sanna*
prediktion upp som en procentuell förändring (notera att modellen predikterar en %-förändring mellan den senaste observerade
candlestickens <code>ClosePrice</code> och den *nästa* "osläppta" candlesticken), samtidigt som hela den nuvarande sekvensen
(<code>latest_sequence - latest_sequence[-1]</code>) bortsett från den senaste candlestick (den som predikteras), ställs upp som
y-vektorn. Här körs alltså en enda MSE-gradientfunktion på endast en sekvens av 288 candlesticks, alltså en batch (för 
den färdigkompilerade modellen som finns i <code>inference/files/</code>), innan modellen gör sin prediktion. Se <code>inference/model_manager.py</code>
för precisa detaljer kring detta. Notera att den färdigkompilerade modellen är tränad med MSE som intern förlustfunktion, varpå 
<code>keras.train_on_batch(X,Y)</code> använder detta för att uppdatera vikterna.

Modellen är med dessa konfigurationer tränad på cirka 1 miljon olika candlesticks, där dessa under förbehandling grupperas i deras symboler (för att få rätt skalning).
Notera att en <code>Dictionary</code> exporteras till <code>inference</code> som innehåller en <code>MinMaxScaler</code> för var och en
av dessa grupperingar.

<b>Notera att Binances API kan ändras, och man kan i <code>fetching</code> få annat format på datat än vad koden är utformad för.</b>


### Kom igång med testkörning

Om man endast vill prova göra prediktioner, 
så behöver man endast <code>pip install binance-connector</code>
för själva websocket-strömmen, där valfri IDE borde kunna sköta installationen av resterande moduler (jag använder PYCHARM community edition) <br><br>
Om du vill träna en ny modell behöver du sätta upp en databastabell enligt schemat i <code>training/db_fetcher</code>, samt <code>pip install python-binance</code>, <code>pip install keras-tuner</code> (eller <code>pip install bayesian-optimization</code>). I detta fall, kör först <code>fetching</code>, och sedan <code>training</code>.

Jag har <u>inte</u> lagt upp någon <code>requirements.txt</code> eller dylikt, så man får manuellt gå igenom beroenden (vilket inte är så farligt, är mest tensorflow och numpy)


###