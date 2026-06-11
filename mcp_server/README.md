# PyCrediTools MCP Server

Servidor MCP para usar o `pycreditools` com agentes de credito em cima de dados reais de clientes.

## Entradas suportadas

As ferramentas leem arquivos reais informados por caminho:

- CSV: `.csv`
- Excel: `.xlsx`, `.xls`, `.xlsm`
- Parquet: `.parquet`

Para Excel, use `sheet_name` quando a planilha tiver mais de uma aba.

## Instalacao

No diretorio do pacote `pycreditools`:

```bash
cd pycreditools
pip install -e ".[mcp]"
```

## Execucao local

Na raiz do repositorio `agentecredito`:

```bash
python -m mcp_server.server
```

## Tools disponiveis

- `inspect_credit_data`: inspeciona colunas, tipos, valores ausentes, abas e amostra do arquivo.
- `simulate_credit_policy`: executa uma `CreditPolicy` serializada contra CSV/Excel/Parquet.
- `predict_credit_decision`: aplica regras de producao exportadas por `DeploymentPolicy`.
- `optimize_score_cutoffs`: otimiza cortes de score usando a base historica do cliente.
- `fit_risk_ratings`: cria ratings de risco a partir de scores e default historico.
- `export_production_rules`: exporta JSON limpo para motor de decisao.
- `explain_policy`: descreve uma politica serializada em texto.

## Fluxo recomendado para agente de credito

1. Chamar `inspect_credit_data` no arquivo do cliente.
2. Validar se existem as colunas esperadas pela politica.
3. Rodar `simulate_credit_policy` ou `predict_credit_decision`.
4. Para estudos, rodar `optimize_score_cutoffs` e `fit_risk_ratings`.
5. Exportar regras finais com `export_production_rules`.

## Observacoes de seguranca

Este MCP espera receber politicas ja serializadas pelo `pycreditools` ou regras de producao.
Evite aceitar codigo Python arbitrario, callables dinamicos ou expressoes montadas livremente por usuarios finais.
Para dados sensiveis de credito, execute o servidor em ambiente controlado e salve artefatos apenas em diretorios autorizados.
