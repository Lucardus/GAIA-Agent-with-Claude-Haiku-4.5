import os
import gradio as gr
import requests
import inspect
import re
import time
import json
import pandas as pd
from smolagents import CodeAgent, ToolCallingAgent, OpenAIServerModel, DuckDuckGoSearchTool, tool, LiteLLMModel
from dotenv import load_dotenv
load_dotenv()

import spaces

@spaces.GPU
def dummy_gpu_function():
    """Função vazia só para satisfazer o requisito de ZeroGPU do Space."""
    return None

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------

class LiteLLMModelComCache(LiteLLMModel):
    """LiteLLMModel com prompt caching da Anthropic ativado.
 
    Marca a mensagem de sistema (system prompt + tool defs, que o CodeAgent
    reenvia inteiros a cada step) com cache_control. Na 1a chamada esse
    prefixo é escrito no cache (custa um pouco mais); nas chamadas seguintes
    dentro do TTL, ele é lido do cache a ~10% do preço de input. Só tem
    efeito com modelos anthropic/*; para outros provedores o campo é
    ignorado silenciosamente pelo litellm.
    """
 
    def generate(self, messages, *args, **kwargs):
        if messages:
            first_msg = messages[0]
            
            # Detecta de forma segura a role se for dicionário ou objeto
            role = None
            if isinstance(first_msg, dict):
                role = first_msg.get("role")
            elif hasattr(first_msg, "role"):
                role = getattr(first_msg, "role")

            if role == "system":
                conteudo = None
                if isinstance(first_msg, dict):
                    conteudo = first_msg.get("content")
                elif hasattr(first_msg, "content"):
                    conteudo = getattr(first_msg, "content")

                if isinstance(conteudo, str):
                    conteudo = [{"type": "text", "text": conteudo}]
                
                if isinstance(conteudo, list) and conteudo:
                    conteudo[-1] = {**conteudo[-1], "cache_control": {"type": "ephemeral"}}
                    
                    if isinstance(first_msg, dict):
                        messages[0]["content"] = conteudo
                    elif hasattr(first_msg, "content"):
                        setattr(messages[0], "content", conteudo)
                        
        return super().generate(messages, *args, **kwargs)

model = LiteLLMModelComCache(
    model_id="anthropic/claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0.1,
    max_tokens=8000
)
 
search_tool = DuckDuckGoSearchTool()

@tool
def baixar_arquivo(task_id: str) -> str:
    """Baixa o arquivo anexo da pergunta usando o Task ID fornecido pela plataforma.
    Use SEMPRE que a pergunta mencionar arquivos locais, anexos, planilhas (.csv, .xlsx),
    PDFs, imagens ou áudios (.mp3).
 
    Args:
        task_id: O ID da tarefa atual (ex: 'task_0', 'task_1').
    """
    # Endpoint correto da API do curso: GET /files/{task_id}
    url = f"{DEFAULT_API_URL}/files/{task_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
 
        cd = resp.headers.get("content-disposition")
        if cd and "filename=" in cd:
            match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
            nome_arquivo = match.group(1) if match else f"arquivo_{task_id}.tmp"
        else:
            nome_arquivo = f"arquivo_{task_id}.tmp"
 
        with open(nome_arquivo, "wb") as f:
            f.write(resp.content)
        return nome_arquivo
    except Exception as e:
        return f"Erro ao baixar o arquivo: {e}"
 
 
@tool
def visitar_pagina(url: str) -> str:
    """Visita uma página web e retorna seu conteúdo em texto.
 
    Args:
        url: o endereço da página a ser visitada.
    """
    from markdownify import markdownify
 
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        texto = markdownify(resp.text)
        limite = 6000
        if len(texto) > limite:
            return texto[:limite] + "\n\n[...conteúdo truncado...]"
        return texto
    except Exception as e:
        return f"Erro ao acessar a página: {e}"
 
 
@tool
def data_atual() -> str:
    """Retorna a data e hora atual."""
    from datetime import datetime
    return datetime.now().isoformat()
 
 
@tool
def transcrever_audio(caminho: str) -> str:
    """Transcreve um arquivo de áudio (mp3/wav) para texto.
 
    Args:
        caminho: caminho local do arquivo de áudio.
    """
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    try:
        with open(caminho, "rb") as f:
            transcricao = client.audio.transcriptions.create(
                file=f, model="whisper-large-v3"
            )
        return transcricao.text
    except Exception as e:
        return f"Erro ao transcrever o áudio: {e}"
 
 
@tool
def buscar_e_resumir(query: str) -> str:
    """Busca na web e retorna um resumo do resultado mais relevante.
    Usa Tavily se houver TAVILY_API_KEY configurada; caso contrário cai
    para DuckDuckGo.
 
    Args:
        query: termo de busca.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_key,
                    "query": query,
                    "max_results": 5,
                    "include_answer": True,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            partes = []
            if data.get("answer"):
                partes.append(f"Resposta resumida: {data['answer']}")
            for r in data.get("results", [])[:5]:
                partes.append(f"- {r.get('title')}: {r.get('content', '')[:500]} ({r.get('url')})")
            texto = "\n".join(partes)
            if texto:
                return texto[:4000]
        except Exception:
            pass
 
    resultado = search_tool(query)
    return str(resultado)[:4000]
 
 
@tool
def ler_planilha(caminho: str) -> str:
    """Lê um arquivo CSV ou Excel e retorna um resumo estruturado.
 
    Args:
        caminho: caminho local do arquivo baixado.
    """
    try:
        if caminho.endswith(".csv"):
            df = pd.read_csv(caminho)
        else:
            df = pd.read_excel(caminho)
    except Exception as e:
        return f"Erro ao ler o arquivo: {e}"
 
    partes = [
        f"Formato: {df.shape[0]} linhas x {df.shape[1]} colunas",
        f"Colunas: {list(df.columns)}",
        f"Tipos:\n{df.dtypes.to_string()}",
    ]
 
    if df.shape[0] <= 50:
        partes.append(f"Dados completos:\n{df.to_string()}")
    else:
        partes.append(f"Primeiras linhas (Head):\n{df.head(5).to_string()}")
        partes.append(f"Últimas linhas (Tail):\n{df.tail(5).to_string()}")
        try:
            partes.append(f"Estatísticas numéricas:\n{df.describe().to_string()}")
        except Exception:
            pass
        partes.append(
            "Planilha grande: use pandas no seu código pra filtrar/inspecionar "
            "linhas específicas em vez de depender só deste resumo (ex.: "
            "print(df[df['coluna'] == 'valor']))."
        )
 
    return "\n\n".join(partes)
 
@tool
def ler_pdf(caminho: str) -> str:
    """Extrai o texto de um arquivo PDF.
 
    Args:
        caminho: caminho local do arquivo PDF baixado.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return "Erro: biblioteca pypdf não instalada."
 
    try:
        reader = PdfReader(caminho)
        textos = []
        for i, page in enumerate(reader.pages):
            texto_pagina = page.extract_text() or ""
            textos.append(f"--- Página {i + 1} ---\n{texto_pagina}")
        texto_final = "\n".join(textos)
        if len(texto_final) > 6000:
            return texto_final[:6000] + "\n\n[...PDF truncado...]"
        return texto_final if texto_final.strip() else "PDF sem texto extraível (pode ser escaneado/imagem)."
    except Exception as e:
        return f"Erro ao ler o PDF: {e}"
 
 
@tool
def ler_imagem(caminho: str, pergunta: str = "Descreva esta imagem em detalhes, incluindo qualquer texto visível.") -> str:
    """Analisa uma imagem (gráfico, captura de tela, foto, tabela escaneada etc.)
    usando um modelo com visão e retorna a descrição/texto extraído.
 
    Args:
        caminho: caminho local da imagem baixada.
        pergunta: o que perguntar sobre a imagem.
    """
    import base64
    import litellm
 
    try:
        with open(caminho, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
 
        ext = caminho.split(".")[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
 
        resposta = litellm.completion(
            model="anthropic/claude-haiku-4-5-20251001",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": pergunta},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        )
        return resposta.choices[0].message.content
    except Exception as e:
        return f"Erro ao analisar a imagem: {e}"
 
 
def validar_formato_resposta(resposta: str) -> str:
    """Limpa a resposta final para bater com as regras de formato do GAIA."""
    r = resposta.strip().strip('"\'` ')
    r = re.sub(r'^(a resposta é|resposta|answer|final answer)[:\s]*', '', r, flags=re.IGNORECASE)
    if r.endswith(".") and not re.match(r'.*\d\.$', r):
        r = r[:-1]
    return r.strip()
 
 
def montar_prompt(pergunta: str, task_id: str) -> str:
    contexto_task = f"Você está resolvendo a tarefa com Task ID: '{task_id}'.\n" if task_id else ""
    return f"""{contexto_task}Pergunta original do usuário: {pergunta}
 
    REGRAS DE FORMATO DE RESPOSTA (GAIA) — siga EXATAMENTE, pois a avaliação é por correspondência exata:
    - Números: escreva apenas o número (sem separador de milhar, sem $ ou % salvo se pedido, sem texto ao redor).
    - Strings: não use artigos (a, o, um) nem abreviações; escreva por extenso os números dentro da string.
    - Listas: separe por vírgula, sem "e" antes do último item.
    - NUNCA inclua frases como "a resposta é" ou explicações no valor final.
    - Ao final, chame a tool final_answer(resposta) com APENAS o valor cru, sem frases.
 
    FERRAMENTAS DISPONÍVEIS PARA ARQUIVOS:
    - baixar_arquivo: baixa o anexo da tarefa (se houver).
    - ler_planilha: para .csv/.xlsx
    - ler_pdf: para .pdf
    - ler_imagem: para .png/.jpg/.jpeg/.gif/.webp
    - transcrever_audio: para .mp3/.wav
 
Resolva o problema passo a passo usando código Python válido."""
 
 
agent = CodeAgent(
    model=model,
    tools=[
        visitar_pagina,
        ler_planilha,
        ler_pdf,
        ler_imagem,
        data_atual,
        buscar_e_resumir,
        transcrever_audio,
        baixar_arquivo,
    ],
    add_base_tools=True,
    max_steps=12, 
    additional_authorized_imports=[
        "pandas",
        "numpy",
        "requests",
        "bs4",
        "math",
        "datetime",
        "re",
        "json",
    ],
)
 
 
def responder(pergunta: str, task_id: str = None) -> str:
    if not task_id:
        match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', pergunta)
        if match:
            task_id = match.group(1)
        else:
            match_simples = re.search(r'(task_\d+)', pergunta)
            if match_simples:
                task_id = match_simples.group(1)
 
    prompt = montar_prompt(pergunta, task_id)
 
    max_tentativas = 3
    for tentativa in range(max_tentativas):
        try:
            resultado = agent.run(prompt)
            return validar_formato_resposta(str(resultado))
        except Exception as e:
            erro_str = str(e).lower()
            if "rate limit" in erro_str or "429" in erro_str:
                espera = 20 * (tentativa + 1)  # 20s, 40s, 60s
                print(f"Rate limit atingido, esperando {espera}s...")
                time.sleep(espera)
            elif tentativa < max_tentativas - 1:
                print(f"Tentativa {tentativa + 1} falhou: {e}")
                time.sleep(5)
            else:
                print(f"Erro ao processar pergunta após {max_tentativas} tentativas: {e}")
                return "Não foi possível determinar a resposta"
    return "Não foi possível determinar a resposta"
 
 
def rodar_todas_perguntas(perguntas: list) -> list:
    """Roda o agente em uma lista de perguntas, salvando incrementalmente
    em resultados_parciais.json a cada resposta (evita perder progresso
    se o script travar no meio do batch).
 
    Args:
        perguntas: lista de dicts com 'question' e opcionalmente 'task_id'.
    """
    resultados = []
    for i, q in enumerate(perguntas):
        print(f"Processando pergunta {i + 1}/{len(perguntas)}...")
        resp = responder(q["question"], q.get("task_id"))
        resultados.append({"task_id": q.get("task_id"), "answer": resp})
 
        with open("resultados_parciais.json", "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
 
        time.sleep(2)
 
    return resultados

class BasicAgent:
    def __init__(self):
        print("Agent initialized com Groq + smolagents.")
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        answer = responder(question)
        print(f"Agent returning answer: {answer[:100]}...")
        return answer

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(ssr_mode=False, server_name="0.0.0.0", server_port=7860)