# Agente de Credito MCP

Agente de credito baseado em MCP para inspecionar bases reais de clientes, simular politicas de credito, aplicar regras de decisao em producao, otimizar cortes de score e gerar ratings de risco usando o runtime `pycreditools`.

Este repositorio separa duas responsabilidades:

- `mcp_server/`: camada MCP que expoe ferramentas para o agente.
- `pycreditools/`: biblioteca Python com a logica de negocio, simulacao, otimizacao e regras de credito.

## O que e MCP

MCP, ou Model Context Protocol, e um protocolo para conectar modelos de linguagem a ferramentas externas de forma padronizada. Em vez de colocar toda a logica dentro do prompt do modelo, o agente chama ferramentas MCP com entradas estruturadas e recebe respostas tambem estruturadas.

Neste projeto, o MCP permite que um agente de credito execute operacoes como:

- inspecionar arquivos CSV, Excel ou Parquet;
- validar colunas e tipos de dados;
- simular uma politica de credito serializada;
- aplicar regras de producao exportadas;
- otimizar cutoffs de score;
- construir ratings de risco;
- exportar artefatos em CSV e JSON.

## Por que a logica nao fica no MCP

O MCP deve ser uma camada fina de integracao, nao o lugar principal da regra de negocio.

Neste projeto, a logica de credito fica em `pycreditools` porque isso traz vantagens praticas:

- Testabilidade: a biblioteca pode ser testada diretamente, sem precisar subir um servidor MCP.
- Reuso: a mesma logica pode ser usada por notebooks, APIs, jobs batch, scripts internos ou pelo MCP.
- Governanca: regras de credito, simulacao e otimizacao ficam versionadas como codigo Python normal.
- Menor acoplamento: trocar o cliente MCP, o modelo de linguagem ou o provedor de LLM nao exige reescrever a regra de credito.
- Seguranca: o MCP apenas recebe parametros estruturados e chama funcoes conhecidas, evitando executar codigo arbitrario vindo do agente.

Em resumo: o MCP e a porta de entrada para o agente; `pycreditools` e o motor de decisao.

## Estrutura

```text
.
├── mcp_server/
│   ├── server.py
│   └── README.md
└── pycreditools/
    ├── pyproject.toml
    ├── README.md
    └── src/pycreditools/
```

## Requisitos

- Python 3.10 ou superior
- Credenciais AWS configuradas, caso o agente seja executado com AWS Bedrock
- Acesso habilitado no Amazon Bedrock ao modelo que sera usado pelo host do agente

## Instalacao local

Clone o repositorio e instale o pacote em modo editavel com as dependencias do MCP:

```bash
git clone https://github.com/elimesquita/agentecredito.git
cd agentecredito
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./pycreditools[mcp]"
```

Rode o servidor MCP:

```bash
python -m mcp_server.server
```

## Configuracao em um host MCP

Adicione o servidor MCP no cliente/host que vai orquestrar o agente. Um exemplo generico de configuracao:

```json
{
  "mcpServers": {
    "agentecredito": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/caminho/para/agentecredito"
    }
  }
}
```

Se estiver usando ambiente virtual, prefira apontar o `command` para o Python da `.venv`:

```json
{
  "mcpServers": {
    "agentecredito": {
      "command": "/caminho/para/agentecredito/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/caminho/para/agentecredito"
    }
  }
}
```

## Uso com AWS Bedrock

O AWS Bedrock entra como provedor do modelo de linguagem. O Bedrock nao substitui o MCP: ele executa o raciocinio do agente, enquanto o MCP fornece as ferramentas que o agente pode chamar.

Fluxo recomendado:

1. Habilite o modelo desejado no Amazon Bedrock.
2. Configure credenciais AWS no ambiente onde o host do agente sera executado.
3. Configure o host do agente para usar Bedrock como provedor de LLM.
4. Configure este projeto como servidor MCP no mesmo host.
5. Instrua o agente a usar as ferramentas MCP para inspecionar dados e executar politicas de credito.

Exemplo de variaveis comuns para o ambiente:

```bash
export AWS_PROFILE=seu-profile
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=seu-modelo-no-bedrock
```

O desenho fica assim:

```text
Usuario
  -> Host do agente
      -> LLM no AWS Bedrock
      -> Servidor MCP agentecredito
          -> pycreditools
          -> arquivos CSV/Excel/Parquet
          -> artefatos CSV/JSON
```

O ponto importante e que a regra de credito nao deve ser delegada ao modelo. O modelo pode decidir qual ferramenta chamar, explicar resultados e conduzir a conversa, mas as decisoes numericas e regras auditaveis devem ser executadas pelo `pycreditools`.

## Ferramentas MCP disponiveis

- `inspect_credit_data`: inspeciona colunas, tipos, valores ausentes, abas e amostra do arquivo.
- `simulate_credit_policy`: executa uma `CreditPolicy` serializada contra CSV, Excel ou Parquet.
- `predict_credit_decision`: aplica regras de producao exportadas por `DeploymentPolicy`.
- `optimize_score_cutoffs`: otimiza cortes de score usando a base historica do cliente.
- `fit_risk_ratings`: cria ratings de risco a partir de scores e default historico.
- `export_production_rules`: exporta JSON para motor de decisao.
- `explain_policy`: descreve uma politica serializada em texto.

## Dados suportados

As ferramentas leem arquivos locais por caminho:

- `.csv`
- `.xlsx`
- `.xls`
- `.xlsm`
- `.parquet`

Para planilhas Excel com mais de uma aba, informe `sheet_name`.

## Seguranca e privacidade

- Nao publique bases reais de clientes no repositorio.
- Nao versione `.env`, chaves AWS, tokens, certificados ou credenciais.
- Execute o MCP em ambiente controlado quando trabalhar com dados sensiveis.
- Restrinja os diretorios onde artefatos podem ser gravados em ambientes produtivos.
- Evite permitir que usuarios finais enviem codigo Python arbitrario ou expressoes dinamicas.

## Licenca

MIT. Veja `LICENSE`.
