import utils
import processingMusic
import classifier
import os

# URL da playlist
playlist_url = input("Qual o endereço da música? : ")
# Diretório de saída
output_dir = "result"

os.makedirs(output_dir, exist_ok=True)
utils.download_music(playlist_url, output_dir)
print("Download concluído!")

utils.convert_mp3_to_wav(output_dir)
print("Conversão concluída!")

name = utils.trim_audio_3_in_parts(output_dir, 15)
print("Dividido em partes!")

processingMusic.criar_dataset('result/data')

# Obtém o nome do arquivo de música na pasta 'result'
for filename in os.listdir(output_dir):
    if filename.endswith(".wav"):  # Verifica se o arquivo é .wav
        print(f"Música: {filename}")
        processingMusic.ExtractToDatabase('result/'+filename, filename, 'none', 'result/data')
        # Remove o arquivo de música após o processamento
        os.remove(os.path.join(output_dir, filename))
classifier.test_genre('result/data.csv', name[7:-4])
