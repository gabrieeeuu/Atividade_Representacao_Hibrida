# Recomendador de Cursos Online

Este projeto utiliza Streamlit para criar um sistema de recomendação de cursos online baseado em técnicas de processamento de texto e agrupamento de usuários.

## Instalação

1. **Clone o repositório ou baixe os arquivos do projeto.**

2. *(Opcional)* Crie um ambiente virtual para o projeto:
   ```sh
   python -m venv .venv
   ```

3. Ative o ambiente virtual:
   - **Windows:**
     ```sh
     .venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```sh
     source .venv/bin/activate
     ```

4. Instale as dependências:
   ```sh
   pip install -r requirements.txt
   ```

## Execução

1. Execute o aplicativo Streamlit:
   ```sh
   streamlit run app.py
   ```

2. O sistema irá baixar automaticamente o arquivo CSV de dados caso não exista na pasta do projeto.

3. Acesse o endereço exibido no terminal para utilizar o sistema de recomendação.

## Observações

- O arquivo de dados será baixado automaticamente na primeira execução.
- Para desativar o ambiente virtual, utilize:
  ```sh
  deactivate