# Inferência da ResNet18 no Fashion-MNIST: *trade-off* e escolha do *batch size*

Código dos experimentos do artigo **"Inferência da ResNet18 no Fashion-MNIST: modelagem do *trade-off* e otimização do *batch size* sob a perspectiva da Pesquisa Operacional"**, apresentado no SBPO 2026.

Hugo César Leal, Andressa Caroline Lopes de Assis, Sadi Martins de Castro Garcia Mendes, Ciniro Aparecido Leite Nametala, Marcos Roberto Ribeiro e Bruno Alberto Soares Oliveira
IFMG – Campus Bambuí · Grupo de Pesquisa em Sistemas Computacionais (GPSisCom)

## Sobre

O trabalho trata o *batch size* como variável de decisão na inferência de uma rede convolucional. A pergunta é qual tamanho de lote dá o melhor compromisso entre produtividade e uso de recursos, considerando que a memória e o hardware disponíveis limitam as opções.

A ResNet18 pré-treinada no ImageNet faz a inferência sobre as 10.000 imagens de teste do Fashion-MNIST, em CPU e em GPU. Para cada configuração são medidos tempo total, *throughput*, *speedup* em relação a um *baseline* sequencial (CPU, lote 1, `num_workers=0`) e eficiência paralela. A fração sequencial é estimada pela Lei de Amdahl.

- CPU: lotes 1 (*baseline*), 8, 32 e 64
- GPU: lotes 1, 8, 32, 64, 128, 256 e 512
- 5 repetições por configuração (média e desvio padrão)

## Ambientes

| Ambiente | CPU | GPU | VRAM | RAM |
|---|---|---|---|---|
| Google Colab | Intel Xeon @ 2.00GHz (2 vCPUs) | Tesla T4 | 16 GB | 12 GB |
| Máquina Local 1 | AMD Ryzen 7 5800H | GeForce GTX 1650 | 4 GB | 16 GB |
| Máquina Local 2 | AMD Ryzen 5 5500 | GeForce RTX 3060 | 8 GB | 32 GB |
| Máquina Local 3 | AMD Ryzen 5 5600 | GeForce RTX 4060 | 8 GB | 16 GB |

## Principais resultados (GPU)

| Ambiente | Faixa eficiente | Menor tempo (s) | Pico de *throughput* (img/s) | *Speedup* máx. | S estimado |
|---|---|---|---|---|---|
| Google Colab | 32 | 11,87 | 842,17 | 50,08 | ≈ 0,38 |
| Máquina Local 1 | 64–128 | 15,97 | 627,71 | 12,85 | ≈ 0,42 |
| Máquina Local 2 | 64–128 | 7,01 | 1.425,70 | 22,20 | ≈ 0,16 |
| Máquina Local 3 | 64–128 | 6,53 | 1.531,57 | 24,83 | ≈ 0,15 |

Em resumo: a GPU supera a CPU em todos os ambientes, mas os ganhos se concentram nos primeiros aumentos de lote e depois passam a ser marginais. O maior lote não é a melhor escolha. A faixa recomendada fica entre 32 e 128 (32 no Colab, 64 a 128 nas máquinas locais). Na Máquina Local 1, com 4 GB de VRAM, o tempo sobe de 15,98 s para 61,20 s no lote 256.

O *speedup* alto do Colab vem de um *baseline* em CPU mais lento (≈ 595 s), e não de produtividade absoluta maior.

Os valores de S acima foram obtidos com o *speedup* intra-GPU (tempo com lote 1 / tempo com lote *b*), ajustado por mínimos quadrados em cada ambiente. O notebook calcula S de forma pontual, com o *speedup* contra o *baseline* em CPU. Por isso o valor impresso lá é diferente (no Colab sai negativo); a explicação está na seção 9 do notebook.

## Arquivos

- `PPD.ipynb`: notebook com todo o experimento. As saídas salvas são da execução no Google Colab.

Ao rodar, o notebook gera:

- `resultados_inferencia.csv`: tabela com médias, desvios, *speedup* e eficiência;
- `grafico_throughput_vs_batch.png`, `grafico_tempo_vs_batch.png`, `grafico_speedup_vs_batch.png`, `grafico_eficiencia_vs_batch.png`.

## Como executar

**Google Colab**

1. Abra o `PPD.ipynb` no Colab.
2. *Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)*.
3. *Executar tudo*.

**Local**

```bash
pip install torch torchvision numpy pandas matplotlib jupyter
```

Para a parte em GPU, instale o PyTorch com suporte a CUDA de acordo com a sua placa (ver [pytorch.org](https://pytorch.org/get-started/locally/)). O Fashion-MNIST é baixado automaticamente para `./data`. As duas últimas células (`nvidia-smi` e `lscpu`) só servem para registrar o hardware e funcionam apenas em Linux.

A parte em CPU é a mais demorada. No Colab, cada repetição do *baseline* levou perto de 10 minutos, e só a CPU somou algumas horas. Se for rodar em ambiente compartilhado, os tempos podem variar um pouco entre execuções.

## Licença

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE).

## Agradecimentos

IFMG, FAPEMIG, CAPES e CNPq.

## Contato

Hugo César Leal – hugocesarleal@gmail.com
