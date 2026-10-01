# NMRfilter no Streamlit: tutorial completo do usuário

**Compare estruturas candidatas com dados experimentais de RMN 2D, examine a classificação e explore as previsões completas de deslocamentos químicos por átomo.**

- **Aplicativo online:** [nmrfilter.streamlit.app](https://nmrfilter.streamlit.app/)
- **Repositório do código:** [RicardoMBorges/NMRfilter_Streamlit](https://github.com/RicardoMBorges/NMRfilter_Streamlit)
- **Arquivos de exemplo:** [Mock data](https://github.com/RicardoMBorges/NMRfilter_Streamlit/tree/main/mock_data)
- **Vídeo:** [Tutorial em vídeo do NMRfilter](https://www.youtube.com/watch?v=pkY-rmvfDdU)

Edição do tutorial: outubro de 2026. Este guia abrange o fluxo de trabalho no Streamlit e as extensões para exportação de previsões atômicas, diagnóstico HOSE e visualização de estruturas. Essas extensões exigem a versão atualizada correspondente do aplicativo; uma implantação anterior pode apresentar menos controles.

Os nomes dos controles, botões, campos e arquivos foram mantidos como aparecem no aplicativo. As explicações estão em português. Nos arquivos de entrada, use ponto como separador decimal, conforme os exemplos.

## Sumário

1. [O que o NMRfilter faz](#1-o-que-o-nmrfilter-faz)
2. [Início rápido](#2-início-rápido)
3. [Prepare os dados experimentais](#3-prepare-os-dados-experimentais)
4. [Prepare as estruturas candidatas e os nomes](#4-prepare-as-estruturas-candidatas-e-os-nomes)
5. [Prepare o arquivo do espectro experimental](#5-prepare-o-arquivo-do-espectro-experimental)
6. [Configure a análise](#6-configure-a-análise)
7. [Execute o fluxo de análise](#7-execute-o-fluxo-de-análise)
8. [Entenda o agrupamento experimental](#8-entenda-o-agrupamento-experimental)
9. [Interprete a classificação](#9-interprete-a-classificação)
10. [Leia os espectros interativos](#10-leia-os-espectros-interativos)
11. [Explore as estruturas e os deslocamentos atômicos](#11-explore-as-estruturas-e-os-deslocamentos-atômicos)
12. [Baixe e localize os resultados](#12-baixe-e-localize-os-resultados)
13. [Entenda as tabelas de previsões atômicas](#13-entenda-as-tabelas-de-previsões-atômicas)
14. [Use o diagnóstico HOSE](#14-use-o-diagnóstico-hose)
15. [Contribua com atribuições experimentais para o nmrshiftdb2](#15-contribua-com-atribuições-experimentais-para-o-nmrshiftdb2)
16. [Avalie a força das evidências para um candidato](#16-avalie-a-força-das-evidências-para-um-candidato)
17. [Solução de problemas](#17-solução-de-problemas)
18. [Reprodutibilidade e apresentação dos resultados](#18-reprodutibilidade-e-apresentação-dos-resultados)
19. [Execução local e notas sobre implantação](#19-execução-local-e-notas-sobre-implantação)
20. [Perguntas frequentes](#20-perguntas-frequentes)
21. [Referências e contato](#21-referências-e-contato)

## 1. O que o NMRfilter faz

O NMRfilter compara correlações de RMN previstas para uma lista de estruturas moleculares com coordenadas de picos experimentais de RMN 2D. A interface Streamlit dá acesso ao fluxo original do NMRfilter v1.5, com gráficos interativos e ferramentas adicionais para examinar as previsões.

O fluxo combina:

1. Conversão dos SMILES dos candidatos em estruturas moleculares.
2. Previsão de deslocamentos químicos usando um banco de referência baseado em códigos HOSE.
3. Simulação das correlações 2D dos candidatos.
4. Organização das correlações experimentais em um grafo e detecção de comunidades.
5. Associação das correlações simuladas aos picos experimentais e classificação dos candidatos.

A interface atualizada também exporta as previsões completas de **deslocamentos atômicos de ¹³C e ¹H**, incluindo átomos ausentes das listas de correlações simuladas de HSQC ou HMBC. Ela pode mostrar essas previsões nas estruturas moleculares e sinalizar átomos com suporte limitado no banco HOSE.

### A pergunta científica

O aplicativo responde: **Quais das estruturas candidatas fornecidas são mais compatíveis com as evidências experimentais de RMN?**

A ferramenta é útil para triagem de candidatos e desreplicação em misturas. Os candidatos podem vir de anotações por espectrometria de massas, bancos de dados, literatura ou listas curadas. A origem dos candidatos não elimina a necessidade de verificar suas estruturas e sua compatibilidade experimental.

Uma boa posição na classificação justifica examinar um candidato com atenção. Ela não corresponde a uma probabilidade de identidade, a um nível de confiança validado ou à comprovação da presença do composto. O NMRfilter não pode recuperar uma estrutura correta que esteja ausente da lista de candidatos, e subestruturas compatíveis podem produzir correspondências para moléculas diferentes.

### O que deve ser enviado

Você fornece uma lista de estruturas e uma lista de **coordenadas de picos experimentais em ppm**. Esse fluxo não exige dados brutos do Bruker e não realiza, por você, processamento dos dados brutos de RMN, correção de fase, correção de linha de base ou seleção de picos.

## 2. Início rápido

Na primeira execução, use os arquivos da pasta **Mock data** do repositório como um conjunto consistente de exemplos. Mantenha juntos os arquivos de candidatos e de nomes correspondentes.

1. Abra o [NMRfilter](https://nmrfilter.streamlit.app/).
2. Expanda a barra lateral, caso esteja recolhida.
3. Preencha **Project name** com um nome descritivo para o projeto.
4. Envie o arquivo em **Candidate structures — SMILES, one per line**.
5. Envie o arquivo em **Measured 2D NMR spectrum — 13C and 1H shifts**.
6. Envie **Candidate names**, se esse arquivo acompanhar o exemplo.
7. Selecione o solvente usado na aquisição experimental.
8. Confira se cada pico experimental tem o tipo de experimento correto. Para um arquivo com apenas duas colunas numéricas, selecione o experimento em **Two-column spectrum interpretation**.
9. Comece com os parâmetros padrão de agrupamento: **tolerância de ¹³C = 0,2 ppm**, **tolerância de ¹H = 0,02 ppm** e **Louvain/RBER resolution = 0,2**.
10. Mantenha **Generate interactive HTML plots** habilitado. O padrão é gerar gráficos dos 10 primeiros candidatos.
11. Clique em **Run NMRfilter** na aba **Analysis and results**, ou na área principal de análise de uma versão anterior.
12. Examine a classificação e, em seguida, inspecione vários dos candidatos mais bem posicionados nos gráficos interativos.
13. Abra **Structures and atomic shifts** para examinar as previsões atômicas.
14. Baixe a classificação, os dados calculados de RMN e o ZIP completo do projeto e dos resultados.

Salve os resultados enquanto a sessão estiver disponível. Uma sessão do navegador não é um arquivo permanente da análise.

## 3. Prepare os dados experimentais

### Evidências recomendadas

HSQC e HMBC fornecem informações complementares:

| Experimento | Informação principal | Papel na avaliação dos candidatos |
| --- | --- | --- |
| HSQC | Correlações entre próton e carbono diretamente ligados | Testa a compatibilidade com os ambientes dos carbonos protonados |
| HMBC | Correlações próton–carbono de maior alcance | Ajuda a distinguir padrões de conectividade e subestruturas candidatas |
| HSQC-TOCSY | Correlações que se estendem pelos sistemas de spins de prótons | Acrescenta evidências quando esse experimento foi adquirido e está habilitado |

Em geral, o HSQC fornece uma conexão mais direta entre um próton e o carbono ao qual ele está ligado. O HMBC acrescenta evidências de conectividade, mas a observação de uma correlação de longo alcance depende do acoplamento, dos parâmetros da sequência de pulsos, da sensibilidade e do ambiente molecular.

### Antes de exportar uma lista de picos

1. Processe e referencie cada espectro no seu programa de RMN.
2. Confira se os espectros de HSQC e HMBC usam referências de deslocamento químico compatíveis.
3. Selecione correlações cruzadas confiáveis. Revise sinais de solvente, ruído, artefatos e seleções duplicadas.
4. Exporte **primeiro o deslocamento de carbono** e **depois o deslocamento de próton**.
5. Preserve o tipo de experimento em cada linha.
6. Use valores em ppm, em vez de índices de pontos, frequências em Hz ou intensidades.

Uma lista de picos bem revisada costuma ser mais útil do que uma lista indiscriminadamente extensa. Ruído intenso e picos duplicados podem alterar a conectividade do grafo e criar correspondências ocasionais.

O aplicativo utiliza pares de coordenadas. A intensidade dos picos não entra na taxa de correspondência nem na classificação descritas aqui. Evite colocar identificadores de picos ou intensidades antes das duas colunas obrigatórias de coordenadas.

## 4. Prepare as estruturas candidatas e os nomes

### Arquivo de estruturas candidatas

Use um arquivo de texto simples `.smi`, em UTF-8, com **um SMILES por linha**. O campo de envio também aceita `.txt` e `.csv`, mas aceitar uma extensão não torna um CSV arbitrário com várias colunas uma lista válida de estruturas.

Exemplo de `candidates.smi`:

```text
CCO
CC(=O)O
COc1ccccc1
```

Esses exemplos ilustram o formato; não constituem um conjunto de demonstração para um espectro específico.

Para preparar uma entrada confiável:

- Não acrescente um cabeçalho de coluna, como `SMILES`.
- Não inclua números de linhas da planilha nem aspas ao redor dos SMILES.
- Evite linhas em branco no meio da lista. Algumas etapas do código original interrompem a leitura em uma linha vazia, embora o exportador atômico ignore linhas vazias.
- Confira valência, carga, aromaticidade, estereoquímica, sais e fragmentos desconectados.
- Remova candidatos duplicados, salvo se houver uma razão específica para mantê-los.
- Guarde uma cópia estável da lista exata que foi enviada.

O exportador atômico atualizado consegue ler um nome após o SMILES na mesma linha. Um arquivo separado de nomes é preferível para manter a nomenclatura consistente entre a classificação original e as exportações mais recentes.

### Arquivo de nomes dos candidatos

Forneça um nome por candidato, na **mesma ordem** do arquivo de SMILES.

Exemplo de `candidate_names.txt`:

```text
Etanol
Ácido acético
Anisol
```

| Posição do candidato | SMILES | Nome |
| --- | --- | --- |
| 1 | `CCO` | Etanol |
| 2 | `CC(=O)O` | Ácido acético |
| 3 | `COc1ccccc1` | Anisol |

Não ordene os nomes independentemente das estruturas. Uma estrutura quimicamente válida associada ao nome errado pode gerar resultados aparentemente plausíveis atribuídos ao composto incorreto.

Na exportação atômica, a escolha do nome segue esta prioridade: arquivo opcional de nomes, nome na mesma linha do SMILES e, por fim, `candidate_N`. Os IDs dos candidatos correspondem à ordem de entrada. **O ID do candidato na entrada e sua posição na classificação são identificadores diferentes.**

## 5. Prepare o arquivo do espectro experimental

O leitor aceita tabulação, vírgula ou ponto e vírgula como delimitadores, além dos rótulos de seção do formato original. Use ponto como separador decimal para obter um formato mais claro e portável.

### Opção A: três colunas com identificação explícita do experimento

Este é o formato mais claro para arquivos com vários experimentos:

```csv
13C,1H,type
55.20,3.75,HSQC
112.40,6.82,HSQC
148.10,6.82,HMBC
130.50,3.75,HMBC
```

As duas primeiras colunas devem conter os deslocamentos de carbono e próton, nessa ordem. A terceira contém o tipo de experimento.

Os tipos reconhecidos são **HMBC**, **HSQC** e **HSQCTOCSY**. O leitor também normaliza formas como `HSQC-TOCSY`.

As coordenadas acima ilustram apenas a sintaxe. Elas não constituem evidências experimentais para as estruturas exemplificadas na Seção 4.

### Opção B: arquivo dividido em seções, no formato original

```text
HMBC
148.10 6.82
130.50 3.75
HSQC
55.20 3.75
112.40 6.82
```

Cada nome de experimento inicia uma seção. As linhas seguintes pertencem àquele experimento até o próximo rótulo de seção. Tabulações podem substituir os espaços desse exemplo.

### Opção C: lista de duas colunas para um único experimento

```csv
13C,1H
55.20,3.75
112.40,6.82
```

Para esse arquivo, selecione **HSQC** em **Two-column spectrum interpretation**. Selecione **HMBC** para uma lista exclusivamente de HMBC ou **HSQC-TOCSY** para esse experimento.

A opção padrão **Reject as ambiguous** impede que um arquivo sem identificação receba silenciosamente o tipo de experimento errado. Rótulos explícitos nas linhas ou nas seções têm prioridade sobre a opção usada para interpretar linhas não identificadas.

**Não misture picos de HSQC e HMBC sem identificação e atribua um único experimento ao arquivo inteiro.** Identifique as linhas ou as seções antes do envio.

### Verificações de formato

| Item | Interpretação correta |
| --- | --- |
| Primeira coordenada | δ¹³C em ppm |
| Segunda coordenada | δ¹H em ppm |
| Terceira coluna, quando presente | Tipo de experimento |
| Cabeçalho | Um cabeçalho descritivo é aceito; as linhas numéricas determinam os dados |
| Colunas extras | Não conte com sua interpretação como metadados |
| Linhas não numéricas rejeitadas | Podem ser ignoradas; confirme no registro de execução a quantidade de picos carregados |

A identificação do experimento é essencial: correlações simuladas de HSQC devem ser comparadas com picos experimentais de HSQC, e o mesmo vale para HMBC e HSQC-TOCSY.

## 6. Configure a análise

### Solvente

A interface fornecida oferece:

- Methanol-D4 (`CD3OD`): metanol deuterado.
- Chloroform-D1 (`CDCl3`): clorofórmio deuterado.
- Dimethylsulphoxide-D6 (`DMSO-D6`): dimetilsulfóxido deuterado.
- Unreported: solvente não informado.

Escolha o solvente usado na aquisição experimental. Se ele não estiver representado, use **Unreported** e registre o solvente real. Essa escolha não elimina as limitações das previsões relacionadas ao solvente.

### Tolerâncias de agrupamento

| Controle | Padrão | Função |
| --- | --- | --- |
| 13C tolerance (ppm) | 0,2 | Agrupa coordenadas experimentais de carbono na construção do grafo de picos |
| 1H tolerance (ppm) | 0,02 | Agrupa coordenadas experimentais de próton na construção do grafo de picos |
| Louvain/RBER resolution | 0,2 | Controla a detecção de comunidades no grafo experimental |

As duas tolerâncias em ppm são usadas no **agrupamento experimental**. Elas não são janelas independentes de aceitação das previsões de ±0,2 ppm para carbono e ±0,02 ppm para próton. O cálculo original de correspondência utiliza outro critério: um custo ponderado das diferenças de coordenadas, descrito na Seção 9.

Tolerâncias maiores podem conectar mais picos e unir grupos que, de outra forma, permaneceriam separados. Tolerâncias menores podem fragmentar grupos. Uma resolução maior na detecção de comunidades geralmente favorece comunidades menores, mas examine o resultado efetivo em vez de pressupor um número fixo de grupos.

Comece com os valores padrão. Ao explorar alternativas, altere um parâmetro de cada vez e avalie se o agrupamento e a interpretação dos candidatos continuam razoáveis. Não escolha configurações apenas porque elas favorecem um candidato desejado.

### Opções de experimento

- **Use HMBC:** habilitado por padrão.
- **Use HSQC-TOCSY:** desabilitado por padrão; habilite quando tiver os dados correspondentes.
- O HSQC faz parte do fluxo padrão.

Mantenha as opções coerentes com os dados enviados. Um experimento ausente pode aparecer como **N/A** ou **Not evaluated**. Isso impede que uma medida ausente seja confundida com um experimento realizado sem correspondências; não demonstra, porém, que a distância global do código original tenha sido recalibrada para a ausência daquele experimento. Considere essa limitação ao examinar análises de um único experimento.

### Use 2 HOSE spheres

Essa opção avançada da simulação original solicita um ambiente local menos extenso na previsão por HOSE. A documentação original descreve o uso de duas esferas em vez do modo padrão de três esferas.

Ela é diferente do **limiar do diagnóstico HOSE**, que filtra os valores de suporte depois da previsão. Alterar o limiar de diagnóstico não executa novamente o simulador.

No exportador atômico examinado, a chamada de previsão é feita separadamente e não lê a opção `dotwobonds`. Portanto, não pressuponha que habilitar essa opção altere a exportação atômica completa da mesma maneira que a simulação 2D original. Para uma análise inicial que compare os resultados atômicos e as correlações, deixe a opção desabilitada.

### Motor de previsão

A interface Streamlit usa o preditor HOSE. A presença de arquivos de `respredict` na distribuição não significa que a interface esteja executando previsões por aprendizagem profunda; esse modo está desabilitado na implementação documentada.

### Opções de gráficos e diagnóstico

| Controle | Finalidade |
| --- | --- |
| Label simulated spectra | Acrescenta rótulos aos espectros simulados do fluxo original; muitos rótulos podem tornar a geração lenta |
| Generate legacy PNG candidate plots | Produz gráficos estáticos opcionais; desabilitado por padrão |
| Generate interactive HTML plots | Produz espectros interativos com Plotly; habilitado por padrão |
| Interactive plots — Top N candidates | Limita a geração de gráficos, não o tamanho da lista de candidatos nem a exportação atômica |
| Plot appearance | Define a opacidade dos marcadores para os próximos gráficos interativos gerados |
| Debug output | Acrescenta informações de diagnóstico à saída da execução |

O número padrão de candidatos para os gráficos é **10**. Reduza-o se a geração estiver lenta. Alterar a opacidade modifica a apresentação, sem alterar a correspondência de picos ou a classificação.

## 7. Execute o fluxo de análise

Clique em **Run NMRfilter** após configurar a barra lateral. O painel de andamento mostra as etapas principais:

1. **Preparing project:** cria a pasta de trabalho da análise e os arquivos de parâmetros.
2. **Converting candidate structures:** converte a lista de estruturas e, na versão atualizada, exporta as previsões atômicas completas e as estruturas do simulador.
3. **Simulating candidate spectra:** gera os registros de correlações previstas do fluxo original.
4. **Clustering measured peaks and ranking candidates:** constrói o grafo experimental, detecta comunidades, associa correlações dos candidatos e gera os resultados.

O tempo de execução depende da quantidade de candidatos, correlações simuladas, picos experimentais, densidade do grafo, configurações de gráficos e recursos disponíveis. Um grafo de picos muito denso pode exigir bem mais tempo do que um pequeno exemplo curado.

Durante a execução:

- Mantenha a página aberta.
- Leia o painel de andamento antes de concluir que uma etapa travou.
- Evite iniciar várias cópias da mesma análise.
- Se o fluxo parar, abra **Run log** e identifique a última etapa concluída.

As previsões atômicas são geradas antes da classificação. Na interface atualizada, **Calculated NMR data** pode continuar disponível para download após uma falha em uma etapa posterior, desde que a saída atômica tenha sido criada. Esse download não significa que a classificação foi concluída.

## 8. Entenda o agrupamento experimental

Cada vértice do grafo representa um pico experimental de RMN 2D. Relações entre coordenadas de carbono ou próton semelhantes conectam os vértices. A detecção de comunidades divide o grafo em grupos de picos relacionados.

Esses grupos são **comunidades de picos experimentais**. Se a entrada contiver apenas HMBC, eles conterão correlações de HMBC. Se HSQC e HMBC forem fornecidos juntos, uma comunidade poderá conter picos dos dois experimentos.

**Um grupo não corresponde automaticamente a um composto, a um sistema de spins ou a um fragmento estrutural validado de forma independente.** Sobreposição e deslocamentos semelhantes podem conectar sinais de constituintes diferentes. Por outro lado, picos fracos ou ausentes podem separar evidências de uma mesma molécula.

O código atual preserva o agrupamento original de coordenadas de carbono e próton, ancorado no primeiro pico de cada grupo, seguido da união de grupos de carbono conectados por grupos de próton. A ordem de entrada pode, portanto, afetar a formação dos grupos iniciais. Preserve a lista exata de picos e a ordem das linhas para fins de reprodutibilidade.

A barra lateral mantém o nome **Louvain/RBER resolution**. Na implementação atualizada das dependências, a detecção de comunidades usa `leidenalg` com uma partição RBER. Ao redigir a seção de métodos, diferencie o nome do controle na interface do algoritmo efetivamente usado na sua versão do programa.

### Por que um grupo muito grande merece atenção

Um grupo grande pode resultar de evidências realmente densas, sobreposição, artefatos, duplicações ou tolerâncias permissivas. Se a maioria dos picos experimentais estiver em uma única comunidade, a componente da classificação baseada na distribuição entre grupos poderá ter pouco poder de discriminação.

Examine no registro de execução a quantidade de picos, os tamanhos dos maiores grupos e o número de arestas do grafo. Revise os dados antes de interpretar um componente grande como evidência de um único composto dominante.

## 9. Interprete a classificação

Leia **distância, desvio padrão e taxas de correspondência por experimento em conjunto**.

| Resultado | Interpretação |
| --- | --- |
| Rank | Posição relativa do candidato segundo a pontuação combinada implementada |
| Distance | Custo normalizado de discrepância entre coordenadas previstas e experimentais; valores menores são favorecidos |
| Standard deviation | Variação normalizada das frações de correspondência experimental entre comunidades; valores maiores são favorecidos na classificação combinada |
| Matching rate | Picos com correspondência aceita divididos pelas correlações simuladas do candidato para o experimento indicado |

### Distance: distância

O cálculo original associa correlações previstas aos picos experimentais por um algoritmo de atribuição. Para tipos de experimento compatíveis, o custo das coordenadas é:

```text
diferença_ponderada = abs(13C_experimental - 13C_previsto)
                    + 10 × abs(1H_experimental - 1H_previsto)

custo_do_par = diferença_ponderada²
```

Tipos de experimento incompatíveis recebem um custo muito alto. A atribuição é global nos conjuntos comparados, em vez de uma busca independente do vizinho mais próximo para cada pico. O custo bruto do candidato é a soma dos custos dos pares atribuídos dividida pelo número de correlações simuladas. Um pico atribuído conta como correspondência aceita quando seu custo é **menor que 9**.

A distância exibida é, em seguida, normalizada em relação aos custos dos candidatos daquela execução.

**Distance = 0.00 não significa deslocamentos químicos idênticos.** Esse valor pode indicar o candidato de menor custo após a normalização, e o número exibido é arredondado. Uma distância baixa com zero correspondências aceitas não sustenta uma identificação.

A distância não é um erro de deslocamento em ppm. Como a normalização depende da lista de candidatos, valores de execuções distintas com candidatos diferentes não podem ser comparados diretamente como medidas absolutas de ajuste.

### Standard deviation: desvio padrão

Para cada comunidade experimental, o programa calcula:

```text
fração_de_correspondência_da_comunidade = picos experimentais com correspondência aceita na comunidade
                                         / total de picos experimentais na comunidade
```

O programa calcula o desvio padrão dessas frações e normaliza o resultado entre os candidatos. Um valor maior indica que as correspondências se distribuem de forma desigual entre as comunidades. A classificação favorece essa concentração porque um candidato presente em uma mistura pode corresponder a apenas parte dos grupos experimentais.

Esse valor:

- Não representa incerteza de deslocamento químico em ppm.
- Não descreve variação entre espectros replicados.
- Não fornece intervalo de confiança nem probabilidade de identidade.
- Pode ser **N/A** quando não houver variação suficiente para a normalização.

Portanto, os grupos usados nesse cálculo não são exclusivamente de HMBC, a menos que a entrada experimental seja exclusivamente de HMBC.

### Classificação combinada

Quando os dois componentes apresentam uma faixa utilizável, a pontuação combinada do código original é:

```text
pontuação_combinada = [distância_normalizada + (1 - desvio_padrão_normalizado)] / 2
```

Pontuações combinadas menores aparecem primeiro. Quando o desvio padrão não pode ser normalizado, a implementação usa uma alternativa baseada na distância normalizada e em um termo constante. As taxas de correspondência por experimento são resultados complementares importantes; não são probabilidades independentes de identificação.

### Matching rate: taxa de correspondência

Para cada experimento disponível:

```text
taxa_de_correspondência = picos com correspondência aceita / correlações simuladas do candidato
```

**5/6 significa que cinco das seis correlações simuladas do candidato tiveram correspondência aceita.** Não significa que cinco de seis picos experimentais foram explicados.

Exemplos de interpretação:

| Resultado ilustrativo | Significado |
| --- | --- |
| HSQC: 5/6, 83,3% | Cinco correspondências aceitas para seis correlações simuladas de HSQC |
| HMBC: 8/20, 40,0% | Oito correspondências aceitas para vinte correlações simuladas de HMBC |
| HMBC: N/A | Não foram fornecidas evidências experimentais de HMBC para avaliação |
| HMBC: 0/20 | Foram fornecidos dados de HMBC, mas nenhuma correspondência aceita foi relatada para vinte correlações simuladas |

Considere sempre o denominador. Um candidato com 2/2 correspondências tem menos evidências de correlação do que um candidato com 20/22, apesar da porcentagem maior.

Correlações simuladas não correspondem necessariamente a ressonâncias únicas resolvidas experimentalmente. Equivalência e sobreposição podem complicar a relação entre as contagens e o número visível de picos.

## 10. Leia os espectros interativos

Expanda um candidato em **Interactive candidate plots**. HMBC e HSQC aparecem lado a lado quando habilitados; HSQC-TOCSY pode acrescentar um painel. Os eixos de carbono ficam alinhados para facilitar a comparação entre regiões.

O eixo horizontal é **δ¹H**, e o vertical é **δ¹³C**. Ambos seguem a direção invertida usual na apresentação de espectros de RMN.

### Legenda dos marcadores

| Marcador | Significado |
| --- | --- |
| Círculo verde preenchido — Matched | Pico experimental aceito como correspondência para esse candidato |
| Círculo cinza vazio — Simulated | Correlação prevista do candidato; essa série inclui correlações previstas com e sem correspondência |
| Quadrado vermelho vazio — Unmatched | Pico experimental selecionado na atribuição, mas que não satisfez o critério de custo para ser aceito |
| Quadrado cinza vazio — Unused | Pico experimental fora dos conjuntos de correspondências aceitas e de picos atribuídos sem correspondência aceita desse candidato |

**Unused não significa ruído.** Em uma mistura, um pico real de outro constituinte pode não ser utilizado para o candidato exibido. Um pico sem correspondência aceita também não comprova, automaticamente, que a estrutura esteja errada; é preciso considerar limitações das previsões, condições experimentais, sobreposição e dados incompletos.

### Inspeção interativa

- Passe o cursor sobre um marcador para ler as coordenadas, o experimento e a categoria do marcador.
- Amplie uma região aromática, olefínica, oxigenada ou alifática.
- Use os controles do Plotly para deslocar a visualização, restaurar a vista e exportar a figura.
- Clique nos itens da legenda para ocultar ou mostrar séries.
- Baixe o **HTML** independente para examinar os resultados fora do aplicativo.

A visualização inicial destaca aproximadamente 0–10 ppm para próton e 0–200 ppm para carbono. Sinais fora desses intervalos exigem ajuste da visualização; estar fora dos eixos iniciais não significa estar ausente dos dados.

O gráfico interativo usa os mesmos conjuntos de picos com e sem correspondência da classificação. Assim, um ponto verde representa a decisão do algoritmo, e não uma segunda validação independente.

### Controles de opacidade

Os valores padrão destacam as correspondências aceitas e mantêm as demais evidências visíveis:

| Categoria | Opacidade padrão |
| --- | --- |
| Matched | 0,95 |
| Simulated | 0,45 |
| Unmatched | 0,28 |
| Miscellaneous / unused | 0,12 |

Alterar os controles deslizantes define valores para os próximos gráficos gerados. Isso não modifica um HTML que já foi baixado. Aumente a opacidade de **Unused** ao examinar as evidências mais amplas da mistura.

## 11. Explore as estruturas e os deslocamentos atômicos

Abra **Structures and atomic shifts** após executar a versão atualizada.

1. Selecione um candidato em **Compound**. A lista segue os IDs de entrada e inclui o estado da entrada.
2. Escolha **13C** ou **1H** em **Label shifts**.
3. Ative ou desative **Show explicit hydrogens** conforme necessário. Os rótulos de hidrogênio são mais fáceis de examinar quando os hidrogênios explícitos estão visíveis.
4. Ative ou desative **Show simulator atom indices**.
5. Defina **Flag HOSE spheres below**.
6. Selecione uma linha na tabela atômica para destacar o átomo correspondente na estrutura.

### Cores da estrutura

| Cor | Significado |
| --- | --- |
| Âmbar | Previsão utilizável com contagem de esferas HOSE abaixo do limiar escolhido |
| Vermelho | Ausência de previsão utilizável ou de valor de suporte HOSE utilizável |
| Azul | Átomo selecionado na tabela; a seleção pode substituir sua cor de diagnóstico |

Os rótulos mostram deslocamentos químicos simulados em ppm. O desenho pode arredondar os valores para duas casas decimais; consulte a tabela exportada para obter o valor armazenado.

### Índices dos átomos e mapeamento seguro

Os índices pertencem ao simulador após a preparação molecular e a inclusão de hidrogênios explícitos. São **índices internos que começam em 1**, não uma numeração química convencional nem índices garantidamente iguais aos de um leitor externo de SMILES.

O visualizador utiliza a molécula salva pelo simulador e um mapa de átomos, com verificações de identidade, elementos e coordenadas. Ocultar hidrogênios é uma operação de apresentação; não redefine os índices originais.

Não reconstrua atribuições colando o SMILES em outro programa e supondo que o átomo número 7 desse programa seja o átomo número 7 do simulador. Use a molécula e o mapa de átomos exportados pelo simulador.

### Downloads da estrutura

- **Download annotated structure (.svg):** desenho vetorial com os rótulos e destaques selecionados.
- **Download simulator structure (.mol):** representação molecular salva e utilizada no mapeamento.
- **Download atom map (.csv):** identidade do candidato, índice do átomo no simulador, ID do átomo, elemento e coordenadas.

Resultados anteriores sem estruturas salvas pelo simulador não podem ser mapeados com segurança apenas a partir do SMILES. Execute novamente a versão atualizada para gerar os arquivos necessários.

## 12. Baixe e localize os resultados

### Botões de download

| Botão | Conteúdo e finalidade |
| --- | --- |
| Download ranking table (.tsv) | Classificação dos candidatos, indicadores exibidos e resultados de correspondência por experimento |
| Download HTML | Comparação espectral interativa de um candidato |
| Download all interactive plots (.zip) | Gráficos HTML gerados para os primeiros N candidatos |
| Download calculated NMR data (.zip) | Todas as entradas, previsões atômicas completas, correlações organizadas, estados das entradas e saídas originais das previsões |
| Download complete project/results ZIP | Pasta de trabalho da análise, resultados numéricos, gráficos gerados e dados calculados de RMN para exportação |
| Download compound diagnostic (.csv) | Resumo atual do diagnóstico HOSE por composto |
| Download flagged atoms (.csv) | Átomos sinalizados pelo filtro de núcleo e limiar atuais |

O **limite dos primeiros N candidatos nos gráficos não restringe a exportação atômica aos candidatos mais bem classificados**.

### Arquivos dentro do ZIP dos dados calculados de RMN

| Arquivo ou localização | Conteúdo |
| --- | --- |
| `resultprediction.csv` | No ZIP atualizado dos dados calculados, tabela completa de previsões atômicas, com cabeçalhos e identidade do candidato |
| `atomic_predictions.csv` | Os mesmos dados atômicos completos, com um nome de arquivo explícito |
| `atomic_prediction_status.csv` | Um registro de estado para cada entrada não vazia |
| `entries_without_prediction.csv` | Entradas cujo estado não é integralmente `predicted`, incluindo previsões parciais e erros |
| `calculated_correlations.csv` | Correlações 2D simuladas, com nomes de colunas em inglês |
| `organized_correlations.csv` | Correlações 2D organizadas, com cabeçalhos descritivos em português mantidos por compatibilidade |
| `candidate_index.csv` | Identidades dos candidatos, contagens de correlações por experimento e estado das previsões atômicas |
| `candidates/` | Uma pasta numerada por candidato, contendo tabelas atômicas e por experimento |
| `simulator_structures/` | Estruturas `.mol` salvas e mapas de átomos, quando gerados com sucesso |
| `original/` | Saídas não modificadas do simulador e parâmetros efetivos |
| `README.txt` | Descrições específicas da exportação e informações sobre sua completude |

Cada pasta de candidato contém `atomic_predictions.csv`, `13C.csv`, `1H.csv`, `HSQC.csv`, `HMBC.csv`, `HSQCTOCSY.csv` e `structure.smi`.

### A distinção importante entre os arquivos `resultprediction.csv`

O arquivo bruto do simulador original e a exportação organizada usam o mesmo nome em locais diferentes:

- **`resultprediction.csv` na raiz do ZIP atualizado dos dados calculados:** previsões atômicas completas.
- **`original/resultprediction.csv`:** registros brutos de correlações 2D do simulador original, com códigos de experimento e separadores entre candidatos.
- **`calculated_correlations.csv`:** tabela organizada de correlações 2D, com cabeçalhos em inglês.

Versões anteriores da exportação usavam o arquivo na raiz para correlações organizadas. Confira o `README.txt` do ZIP e os cabeçalhos antes de assumir o significado do arquivo.

No ZIP completo do projeto, os dados calculados também ficam reunidos em `calculated_NMR/`. O arquivo de trabalho `result/resultprediction.csv` continua sendo a saída da simulação original; não o confunda com `calculated_NMR/resultprediction.csv`.

### Códigos internos dos experimentos

| Código original | Experimento |
| --- | --- |
| `q` | HSQC |
| `b` | HMBC |
| `t` | HSQC-TOCSY |

As tabelas organizadas de correlações substituem esses códigos por nomes legíveis dos experimentos.

As exportações atômicas e de correlações preservam a **ordem de entrada**, não a ordem da classificação. Combine as tabelas pela identidade do candidato e confira o SMILES; não use apenas a posição da linha exibida.

## 13. Entenda as tabelas de previsões atômicas

As previsões atômicas completas são obtidas diretamente da ferramenta de previsão HOSE. Elas não são reconstruídas a partir das listas de HMBC ou HSQC.

Isso é importante porque um carbono sem hidrogênio ligado não terá correlação direta de HSQC. Uma lista de correlações 2D, por si só, não fornece uma tabela independente completa de todos os átomos de carbono e hidrogênio previstos.

### Campos da tabela atômica

| Coluna | Significado |
| --- | --- |
| `candidate_id` | Identificador do candidato, começando em 1, na sequência de entradas não vazias |
| `input_line` | Número da linha original no arquivo de estruturas |
| `candidate_name` | Nome atribuído ao candidato |
| `smiles` | Representação estrutural fornecida |
| `nucleus` | `13C` ou `1H` |
| `atom_index` | Índice do simulador, começando em 1, após preparação molecular e inclusão de hidrogênios explícitos |
| `atom_id` | Identificador do átomo no simulador/CDK |
| `shift_ppm` | Deslocamento químico médio previsto informado pelo preditor |
| `minimum_ppm` | Valor inferior retornado pelo preditor |
| `maximum_ppm` | Valor superior retornado pelo preditor |
| `hose_spheres` | Contagem de esferas do ambiente local informada para a correspondência com o banco |
| `prediction_status` | Indica se o átomo teve previsão, não teve previsão ou apresentou erro de previsão |
| `message` | Explicação de diagnóstico, quando aplicável |
| `prediction_source` | Origem da previsão |

Os valores mínimo e máximo são saídas do preditor. Não devem ser apresentados como intervalos de confiança estatísticos validados ou como incerteza experimental.

Átomos equivalentes não são consolidados na tabela atômica completa. Três hidrogênios equivalentes de uma metila podem, portanto, produzir três linhas, mesmo que os deslocamentos previstos sejam idênticos. O número de linhas atômicas não equivale ao número de ressonâncias resolvidas nem ao número de picos 2D.

### Valores ausentes e estado das entradas

Um deslocamento químico indisponível fica **em branco**, em vez de receber zero ou um valor sentinela de falha. Um zero numérico real não deve ser usado como substituto de um dado ausente.

| Estado da entrada | Interpretação |
| --- | --- |
| `predicted` | Todos os átomos-alvo exportados receberam previsões utilizáveis |
| `partial_prediction` | Alguns átomos-alvo receberam previsões; outros ficaram indisponíveis ou apresentaram falha |
| `no_prediction` | Nenhum átomo-alvo recebeu uma previsão utilizável |
| `no_target_atoms` | Não foram encontrados átomos-alvo de carbono ou hidrogênio |
| `structure_error` | Houve falha na leitura ou preparação da estrutura |

Toda entrada não vazia tem um registro de estado. Uma estrutura com erro pode não ter linhas atômicas; por isso, use `atomic_prediction_status.csv` para avaliar a completude, em vez de contar apenas as linhas de `atomic_predictions.csv`.

Se os blocos de correlações originais não puderem ser associados inequivocamente às entradas, o exportador preserva a saída bruta e evita atribuir blocos parciais aos candidatos errados. Confira no `README.txt` o indicador **Legacy correlations complete**. Nessa situação, uma tabela organizada de correlações vazia não significa necessariamente que nenhuma correlação bruta tenha sido gerada.

## 14. Use o diagnóstico HOSE

Expanda **HOSE diagnostics — compounds with limited database support** na aba de análise.

Os códigos HOSE descrevem camadas sucessivas do ambiente molecular local de um átomo. Uma correspondência baseada em menos esferas é menos específica estruturalmente do que uma correspondência baseada em mais camadas. Isso permite sinalizar previsões que merecem exame mais cuidadoso.

### Controles

1. Selecione **Flag predictions with fewer than this number of HOSE spheres**.
2. Escolha **Both**, **13C** ou **1H** em **Inspect nucleus**.
3. Examine o resumo por composto.
4. Inspecione as linhas dos átomos sinalizados.
5. Baixe os CSVs de diagnóstico por composto e por átomo.

O limiar padrão é **4**, que sinaliza previsões utilizáveis com contagens de **1–3 esferas**. Previsões ausentes ou inutilizáveis também entram no diagnóstico.

O limiar é uma escolha configurável de triagem. Não é um ponto de corte validado entre previsões corretas e incorretas, e não altera a classificação nem executa novamente as previsões.

### Campos do resumo por composto

| Campo | Significado |
| --- | --- |
| `minimum_hose_spheres` | Menor contagem de suporte utilizável entre os átomos incluídos pelo filtro de núcleo escolhido |
| `low_13C_atoms` | Número de previsões de carbono abaixo do limiar escolhido |
| `low_1H_atoms` | Número de previsões de hidrogênio abaixo do limiar escolhido |
| `atoms_without_hose` | Átomos sem informação utilizável de previsão ou suporte |
| `total_target_atoms` | Número de linhas atômicas incluídas para aquele candidato pelo filtro de núcleo |

As contagens se referem a átomos individuais, incluindo hidrogênios equivalentes. Compare tanto o número quanto a proporção de átomos sinalizados ao avaliar compostos de tamanhos diferentes.

### O que um suporte baixo indica

Um suporte baixo sugere que o preditor encontrou apenas uma correspondência menos específica do ambiente local no banco incluído no programa. Isso não demonstra que:

- A estrutura candidata esteja incorreta.
- O deslocamento previsto seja necessariamente impreciso.
- Os dados experimentais estejam ausentes do banco online atual do nmrshiftdb2.
- Uma previsão com alto suporte seja suficiente para confirmar a identidade do composto.

Use a sinalização para orientar a revisão da literatura, a atribuição experimental e o exame do átomo correspondente na estrutura.

### Quais arquivos do banco são utilizados?

A distribuição examinada inclui as tabelas de referência **`nmrshiftdbc.csv`** e **`nmrshiftdbh.csv`** dentro de **`lib/simulate.jar`**, para previsão de carbono e hidrogênio, respectivamente.

Esses são recursos de referência incluídos no programa. Eles são diferentes de `atomic_predictions.csv`, que contém previsões para as estruturas enviadas na sua análise. Atualizar o banco online ou contribuir com ele não substitui automaticamente os recursos dentro do JAR implantado.

## 15. Contribua com atribuições experimentais para o nmrshiftdb2

O diagnóstico pode identificar compostos para os quais dados experimentais de referência melhores seriam úteis. O aplicativo fornece links para o [nmrshiftdb2](https://nmrshiftdb.nmr.uni-koeln.de/) e suas [instruções de submissão e revisão](https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/using.html).

Um caminho prático é:

1. Identifique o composto e os átomos sinalizados.
2. Procure atribuições experimentais confiáveis em suas próprias medidas ou na literatura.
3. Verifique a estrutura e as atribuições com evidências adequadas.
4. Mapeie cuidadosamente as atribuições para a estrutura submetida; índices do simulador não são numeração química convencional.
5. Inclua solvente, condições relevantes de aquisição e referências das fontes.
6. Siga o processo atual de submissão e revisão do banco. A interface informa que as submissões públicas ficam disponíveis após aprovação por revisores.

**Não submeta deslocamentos simulados como medidas experimentais.** Uma correspondência automatizada em uma mistura não resolvida também não é suficiente, por si só, para estabelecer uma atribuição experimental completa.

O aplicativo utiliza uma cópia do banco de referência. Uma contribuição passa a afetar o preditor local apenas quando os recursos correspondentes forem atualizados na implantação do programa.

## 16. Avalie a força das evidências para um candidato

Para cada candidato bem posicionado, examine:

1. A estrutura é quimicamente plausível para a amostra?
2. O nome fornecido e o SMILES são consistentes?
3. Há correlações aceitas em quantidade suficiente para sustentar uma interpretação relevante?
4. O HSQC é compatível com os ambientes esperados dos carbonos protonados?
5. O HMBC fornece evidências de conectividade que ajudam a distinguir as alternativas?
6. Regiões previstas importantes estão sem suporte experimental? Isso pode ser explicado pelas condições experimentais?
7. As correspondências se concentram em um subconjunto coerente das comunidades experimentais?
8. Os átomos sinalizados pelo diagnóstico HOSE coincidem com previsões questionáveis ou decisivas?
9. Alternativas estruturalmente próximas explicam as mesmas evidências?
10. A interpretação é estável diante de mudanças razoáveis nos parâmetros da análise?

### Conclusões adequadas

| Observação | Interpretação defensável |
| --- | --- |
| Boa classificação com evidências coerentes de HSQC/HMBC | O candidato merece investigação direcionada |
| Porcentagem alta baseada em poucas correlações | Há compatibilidade, mas as evidências são limitadas |
| Resultados semelhantes para estruturas próximas | A distinção entre os candidatos permanece sem resolução |
| Distância baixa sem correspondências aceitas | O custo relativo, sozinho, não fornece evidência de identificação |
| Ausência de aquisição de HMBC | A conectividade de longo alcance não foi avaliada |
| Muitos átomos com baixo suporte HOSE | O suporte das previsões requer análise mais cuidadosa |

A investigação posterior pode incluir melhoria das atribuições, experimentos adicionais de RMN, comparação com material autêntico, fracionamento e integração com evidências independentes de espectrometria de massas ou de outras técnicas.

As taxas de correspondência do NMRfilter não são estimativas de abundância. Elas não quantificam a concentração de um composto nem sua proporção na mistura.

## 17. Solução de problemas

### Nenhum par de coordenadas 13C/1H pôde ser lido

Confira se as duas primeiras colunas são coordenadas numéricas, se a ordem é carbono–próton e se o delimitador é consistente. Remova IDs de picos antes das coordenadas, unidades anexadas aos números e formatações da planilha. Verifique se há linhas numéricas após o cabeçalho.

### Picos sem identificação do tipo de experimento

Para uma lista de duas colunas de um único experimento, selecione seu tipo em **Two-column spectrum interpretation**. Para dados mistos, acrescente rótulos de seção ou uma terceira coluna `type`. Alterar a caixa de seleção de um experimento não identifica um arquivo de entrada ambíguo.

### Todos os candidatos apresentam zero correspondências

Primeiro, verifique os rótulos dos experimentos, a ordem dos eixos, as unidades e a referenciação dos deslocamentos químicos. Em seguida, confirme se as quantidades esperadas de picos medidos e espectros candidatos foram carregadas. Confira se o arquivo é realmente o espectro desejado e se as estruturas candidatas são adequadas.

Não conclua que a análise funcionou com base em uma distância de 0.00. Se o problema persistir, baixe os arquivos do projeto e examine as previsões brutas, a entrada experimental e o registro de execução.

### Nomes dos candidatos incorretos ou erro de índice durante a classificação

Compare as listas de estruturas e nomes linha a linha. Remova linhas vazias e verifique se o número e a ordem dos nomes correspondem aos candidatos. Confirme se a conversão não falhou para alguma entrada. Não corrija um resultado com nomes trocados apenas renomeando as posições da classificação, sem verificar a identidade estrutural.

### Uma etapa parece travada

Abra o painel de andamento e o registro de execução. Verifique se o programa está agrupando um grafo denso, comparando uma grande lista de candidatos ou gerando rótulos e gráficos.

Teste uma lista menor e curada de candidatos e picos. Desabilite os gráficos PNG opcionais e os rótulos simulados, e reduza o número de gráficos HTML. Mudanças nas tolerâncias de agrupamento devem ter justificativa científica, e não apenas servir para acelerar a execução.

### Imagem de fundo do Bruker indisponível

O fluxo baseado em listas de picos não fornece dados brutos do Bruker para a imagem de fundo. Um aviso de que o caminho de HMBC ou HSQC do Bruker não foi configurado pode se referir à imagem de fundo, e não à comparação numérica. Confira o estado final da execução para determinar se a classificação foi concluída.

### A classificação foi concluída, mas os gráficos interativos não aparecem

Confirme se **Generate interactive HTML plots** estava habilitado antes da execução. Examine o registro e procure `plots_html/` no ZIP completo do projeto. Um erro de geração ou incorporação dos gráficos pode ocorrer após a conclusão da classificação numérica.

### A aba de estruturas solicita uma nova execução

Resultados antigos não contêm a molécula persistida pelo simulador nem o mapa de átomos. Execute a versão atualizada uma vez. O visualizador evita estimar atribuições a partir de um novo desenho gerado pelo SMILES.

### A exportação atômica contém campos em branco

Leia `prediction_status`, `message` e a tabela de estado das entradas. Deslocamentos em branco indicam previsões indisponíveis, não zero ppm. Use o diagnóstico HOSE para localizar os átomos afetados.

### A exportação atômica está presente, mas as correlações estão vazias

Leia o `README.txt` da exportação e os estados das entradas. Os fluxos atômico e de correlações são separados. Uma saída incompleta das correlações originais pode ser preservada apenas em formato bruto para evitar atribuição incorreta aos candidatos.

### O download não abre

Aguarde a conclusão e tente baixar novamente. Confirme se o ZIP tem tamanho maior que zero e abre em um programa de descompactação. Para obter todos os dados atômicos, use **Download calculated NMR data (.zip)**, em vez do TSV da classificação ou do ZIP de gráficos. Abra um arquivo HTML baixado em um navegador.

### Os resultados desaparecem após uma interação com a página

O Streamlit executa novamente a interface quando os controles mudam. As seções mais recentes de download atômico, diagnóstico e estruturas mantêm uma referência ao projeto no estado da sessão, mas algumas visualizações da classificação e dos gráficos são criadas dentro da ação de execução. Salve as saídas prontamente; se necessário, execute novamente com os parâmetros registrados.

### O que enviar ao relatar um problema

Forneça o texto do erro, a última etapa concluída, os parâmetros, as quantidades de entradas e um exemplo reproduzível. Arquivos úteis para diagnóstico incluem:

- `realspectrum.csv` e `testall.smi`.
- `testallnames.txt`, quando utilizado.
- `nmrproc.properties`.
- O arquivo bruto `resultprediction.csv`.
- `atomic_predictions.csv` e `atomic_prediction_status.csv`.
- `cluster.txt`, `clusterslouvain.txt` e os arquivos `smart*.csv` relevantes.
- `result.txt`, `ranking_table.tsv` e o registro de execução.

Um exemplo positivo pequeno e conhecido é especialmente útil para distinguir problemas de leitura, previsão, agrupamento, correspondência e geração de gráficos.

## 18. Reprodutibilidade e apresentação dos resultados

Arquive:

- A lista exata de candidatos e nomes na ordem de entrada.
- O arquivo de picos medidos na ordem original das linhas.
- O solvente de aquisição, os tipos de experimento e as informações de referenciação dos deslocamentos químicos.
- A versão do programa ou o commit do repositório, quando disponível, e a data da análise.
- Todos os parâmetros efetivos, incluindo opções de experimentos e modo HOSE.
- O ZIP completo do projeto e dos resultados, o TSV da classificação, as exportações atômicas e os gráficos usados na interpretação.
- Os limiares HOSE e filtros de núcleo usados nos resumos de diagnóstico exportados.

A implementação examinada não define explicitamente uma semente fixa para a detecção de comunidades. Evite afirmar reprodutibilidade exata entre execuções ou ambientes sem verificá-la.

### Sugestão de texto para os métodos

Substitua os campos entre colchetes pelos valores efetivamente utilizados:

> Estruturas candidatas fornecidas como SMILES foram avaliadas contra listas de picos experimentais de [HSQC/HMBC/HSQC-TOCSY] por meio da interface Streamlit do NMRfilter [versão ou commit; data de acesso]. Os deslocamentos químicos foram previstos utilizando o banco de referência HOSE incluído no programa, com [configuração de solvente]. O agrupamento dos picos experimentais empregou tolerâncias de carbono e próton de [valor] e [valor] ppm, respectivamente, seguido de detecção de comunidades RBER pela implementação fornecida naquela versão, com resolução [valor]. A classificação dos candidatos combinou o custo normalizado de atribuição e a variação normalizada das frações de correspondência experimental entre comunidades. As taxas de correspondência por experimento foram expressas em relação às contagens de correlações simuladas dos candidatos. Os candidatos classificados foram examinados por sobreposições espectrais e diagnóstico das previsões atômicas, sendo tratados como hipóteses para avaliação estrutural posterior.

### Sugestão de texto para os resultados

> O candidato X foi priorizado pelo NMRfilter e apresentou [correspondências/total] correlações de HSQC e [correspondências/total] de HMBC. A inspeção das regiões com correspondência sustentou compatibilidade com [evidências estruturais especificadas]. [Número] previsões atômicas ficaram abaixo do limiar de suporte HOSE escolhido. Esses resultados sustentam a investigação posterior do candidato, mas não estabelecem uma identificação inequívoca.

Informe explicitamente os experimentos ausentes. Não descreva N/A como uma taxa de correspondência de 0%, nem o desvio padrão exibido como incerteza do deslocamento químico.

## 19. Execução local e notas sobre implantação

Estas notas são destinadas a quem mantém uma cópia local ou a implantação online. Elas não são necessárias para usar o aplicativo hospedado.

### Execução local no Windows

O pacote distribuído para Windows inclui um inicializador:

1. Extraia o ZIP completamente.
2. Execute `START_NMRFILTER.bat`.
3. Na primeira utilização, o inicializador prepara o ambiente Conda `nmrfilter`, com as versões de Python e Java especificadas pelo pacote, instala as dependências e inicia o Streamlit.
4. Reutilize o inicializador nas execuções seguintes.

A configuração documentada utiliza Python 3.11 e Java 17. Evite iniciar o aplicativo com um ambiente base não relacionado ou um executável do Streamlit instalado globalmente. Se as dependências se tornarem inconsistentes, o pacote pode incluir `RESET_NMRFILTER_ENV.bat` para recriar o ambiente dedicado.

### Streamlit Community Cloud

O arquivo principal do aplicativo é `app.py`. Os arquivos de dependências e de configuração dos pacotes do sistema fornecem os requisitos de Python e Java. Java, RDKit, SciPy, igraph e as dependências relacionadas ao Leiden precisam estar disponíveis para as etapas correspondentes.

Não mantenha caminhos específicos de uma máquina Windows em uma implantação Linux. Os arquivos de propriedades do Java tratam barras invertidas como caracteres de escape; a interface escreve caminhos com barras normais para evitar caminhos malformados no Windows.

Mantenha pastas de projetos gerados, diretórios temporários de simulação, caches e registros fora dos pacotes de código-fonte. Incluir projetos de análises anteriores pode aumentar significativamente o tamanho de um ZIP de distribuição e misturar resultados gerados com o código do aplicativo. O **ZIP completo dos resultados** deve arquivar um projeto de análise; o **ZIP do aplicativo** deve distribuir o programa.

A presença de recursos opcionais de previsão de grande tamanho não significa que eles sejam usados pelo fluxo HOSE hospedado. Ao implantar o aplicativo, confira os tamanhos reais dos arquivos em relação aos limites atuais de hospedagem e do repositório.

## 20. Perguntas frequentes

### O NMRfilter identifica um composto automaticamente?

Ele classifica os candidatos fornecidos segundo a compatibilidade com as evidências de RMN. A confirmação exige avaliação estrutural adicional.

### Os grupos experimentais são correlações de HMBC?

São grupos de picos experimentais. Eles contêm apenas correlações de HMBC quando a entrada contém apenas HMBC. Dados mistos de HSQC/HMBC podem produzir comunidades com os dois tipos.

### Uma distância de 0.00 significa correspondência exata?

Não. É um valor normalizado e arredondado em relação ao conjunto de candidatos. Confira as correspondências aceitas e as coordenadas reais.

### Um desvio padrão maior significa menor precisão da previsão?

Não. Ele descreve a variação das frações de correspondência entre comunidades experimentais. A classificação combinada favorece valores normalizados maiores.

### 5/6 significa que cinco de seis picos experimentais foram explicados?

Significa cinco correspondências aceitas para seis correlações simuladas do candidato no experimento indicado.

### Posso usar apenas HSQC ou apenas HMBC?

O leitor aceita dados de um único experimento identificado explicitamente. Experimentos ausentes devem ser informados como não avaliados. O custo global do código original ainda requer cautela; a exibição de N/A, por si só, não valida o comportamento da classificação para um único experimento. Evidências combinadas e bem revisadas de HSQC e HMBC são mais informativas para a avaliação estrutural.

### Todas as previsões dos candidatos são exportadas ou apenas as mostradas nos gráficos?

A exportação atualizada dos dados calculados inclui toda entrada não vazia na tabela de estado e todas as linhas atômicas disponíveis. A geração de gráficos tem um limite separado para os primeiros N candidatos.

### Qual arquivo contém os dados completos previstos de ¹³C e ¹H?

Use `atomic_predictions.csv` ou o `resultprediction.csv` na raiz do ZIP atualizado dos dados calculados. Use `atomic_prediction_status.csv` para verificar entradas com falhas ou previsões parciais.

### Os rótulos das estruturas são atribuições experimentais?

Não. Eles representam deslocamentos químicos atômicos simulados. Os marcadores verdes nos gráficos espectrais mostram correspondências aceitas de correlações experimentais, que são outro tipo de resultado.

### Os números exibidos nos átomos são a numeração química convencional?

Não. São índices do simulador. Use a molécula salva e o mapa de átomos para preservar a identidade.

### Um suporte HOSE baixo comprova que o nmrshiftdb2 não contém o composto?

Não. Ele descreve a correspondência com a cópia do banco incluída no programa. O banco online atual pode conter dados adicionais.

### O limiar HOSE altera a classificação?

Não. Ele modifica quais previsões são sinalizadas no diagnóstico e na visualização da estrutura. É diferente da opção de duas esferas da simulação.

### Posso comparar distâncias normalizadas entre listas diferentes de candidatos?

Não como medidas absolutas de ajuste. Acrescentar ou remover candidatos altera a faixa de normalização.

### Posso quantificar a composição da mistura pelas taxas de correspondência?

Não. As taxas descrevem compatibilidade das correlações e não são medidas calibradas de abundância.

## 21. Referências e contato

### Cite a metodologia original do NMRfilter

Kuhn, S., Colreavy-Donnelly, S., de Andrade Silva Quaresma, L. E., et al. (2020). Applying NMR compound identification using NMRfilter to match predicted to experimental data. *Metabolomics*, **16**, 123. [https://doi.org/10.1007/s11306-020-01748-1](https://doi.org/10.1007/s11306-020-01748-1).

### Trabalho relacionado sobre análise integrada de misturas por MS/RMN

Kuhn, S., Colreavy-Donnelly, S., de Souza, J. S., and Borges, R. M. (2019). An integrated approach for mixture analysis using MS and NMR techniques. *Faraday Discussions*, **218**, 339–353.

Ao utilizar a adaptação para Streamlit, registre também o [repositório](https://github.com/RicardoMBorges/NMRfilter_Streamlit), a data de acesso ao aplicativo e a versão do programa ou o commit, quando disponíveis.

### Contato

- **Ricardo M Borges:** [ricardo_mborges@ufrj.br](mailto:ricardo_mborges@ufrj.br)
- **Stefan Kuhn:** [stefan.kuhn@ut.ee](mailto:stefan.kuhn@ut.ee)

Para dúvidas técnicas, inclua o texto do erro, os parâmetros relevantes, as quantidades de entradas e um pequeno exemplo reproduzível sempre que possível.
