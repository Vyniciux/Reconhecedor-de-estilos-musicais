## Reconhecedor de estilos musicais

Escrito em python o reconhecedor extrai características como frequência (trasnformadas de fourier), beats por segundos, Mel-frequency cepstral coefficients, etc, para analisar e classificar músicas (SVC). 
Para rodar o programa é necessário importar os módulos: yt-dlp, soundfile, librosa, pandas e ffmpeg.

`pip install yt-dlp soundfile librosa pandas ffmpeg`

O programa trabalha com as músicas presentes na pasta "data" e cada pasta representa um gênero, o projeto é modularizável, assim você pode adicionar mais músicas na pasta e deixar o classificador mais robusto e eficaz.

<img width="828" height="284" alt="Processando musicas" src="https://github.com/user-attachments/assets/34f98851-339e-4250-8faf-0b55a44baf5b" />

Para usar o programa execute o arquivo run.py e cole no terminal o link (youtube) para música ou playlist que deseja ser reconhecida.

`python run.py`

<img width="961" height="313" alt="Link para musica" src="https://github.com/user-attachments/assets/9315f4f7-e9bd-4626-80d7-dad0bf1e5c76" />

<img width="552" height="260" alt="Resultado" src="https://github.com/user-attachments/assets/8b82843b-440f-417b-ac59-03b7eac37ce4" />

<img width="745" height="254" alt="Outro resultado" src="https://github.com/user-attachments/assets/194a5531-28fc-4010-adfe-8ebd37866e36" />

<img width="755" height="260" alt="Ainda outro resultado" src="https://github.com/user-attachments/assets/4627b373-353b-4180-bfef-968e351b6067" />

Nota: A mistura de gêneros e sons que exite faz com que essa não seja uma tarefa trivial, ainda mais com um número tão pequeno de dados quanto a base desse projeto, então acreditamos que com um pouco mais de dados o projeto consiga ser muito útil.

Desenvolvido por: Juliana Silva, Leandro Vynicius Ramos da Silva e Thyago Barbosa Soares. Para a disciplina de Sinais e sistemas.
