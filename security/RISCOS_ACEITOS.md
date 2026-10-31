# Riscos aceitos - AgroSmart (Fase 7)

O pipeline DevSecOps falha automaticamente quando encontra uma vulnerabilidade nova.
As excecoes abaixo foram analisadas e aceitas de forma explicita, com justificativa e data de revisao.

## Pillow 12.3.0 (39 vulnerabilidades corrigidas apenas na 13.0.0)

| Item | Detalhe |
|---|---|
| Situacao inicial | `pillow==10.2.0` com 62 vulnerabilidades e `pyarrow` sem versao fixa (64 no total, via pip-audit) |
| Acao tomada | Atualizacao para `pillow==12.3.0` e `pyarrow==23.0.1` (eliminou 25 vulnerabilidades) |
| Bloqueio | `streamlit==1.56.0`, a versao atual, exige `pillow<13` |
| Exposicao | Os decodificadores afetados (TIFF, PDF, EPS, GIF, MPO, BMP RLE, WMF...) so sao acionados ao abrir arquivos nesses formatos. O dashboard abre apenas imagens JPG/PNG da pasta local `input_images/`, com caminho validado contra *path traversal*. Nao ha upload de arquivos por usuarios. |
| Mitigacao | Containers executam como usuario nao-root; o dashboard nao recebe arquivos externos. |
| Monitoramento | O pip-audit e o Trivy continuam listando essas IDs nos relatorios a cada execucao do pipeline. |
| Revisao | Atualizar para Pillow 13 assim que o Streamlit liberar compatibilidade; remover as IDs de `security/pip-audit-ignore.txt` e `.trivyignore`. |

Arquivos de excecao: `security/pip-audit-ignore.txt` (pip-audit) e `.trivyignore` (Trivy).
