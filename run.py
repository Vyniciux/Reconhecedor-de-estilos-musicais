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

utils.trim_all_folder(output_dir)
print("Corte concluído!")

processingMusic.criar_dataset('result/data')

# Obtém o nome do arquivo de música na pasta 'result'
music_file = next((f for f in os.listdir(output_dir) if f.endswith('.wav')), None)
if music_file:
    print(f"Música: {music_file}")
    processingMusic.ExtractToDatabase('result/'+music_file, music_file, 'none', 'result/data')
    classifier.test_genre('result/data.csv', music_file)
    # Remove o arquivo de música após o processamento
    os.remove(os.path.join(output_dir, music_file))
else:
    print("Nenhuma música encontrada na pasta result.")
