// Feito para a disciplina de Inteligência Artificial
// Simula um diagnóstico médico explorando conexões entre sintomas e doenças
// A busca tem um limite de profundidade, então nem sempre chega até o fim

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NOME 80
#define MAX_CONEXOES 20
#define MAX_DIAGNOSTICOS 50
#define MAX_CAMINHO 10

typedef enum {
    SINTOMA=0,
    CAUSA_INTERMEDIARIA=1,
    DOENCA=2
} TipoNo;

static const char* TIPOS_NO[] = {"Sintoma","Causa","Doenca"};

typedef struct NoMedico NoMedico;

typedef struct {
    NoMedico* destino;
    float probabilidade;
    char descricao[MAX_NOME];
} Conexao;

struct NoMedico {
    char nome[MAX_NOME];
    TipoNo tipo;
    Conexao conexoes[MAX_CONEXOES];
    int num_conexoes;
    int visitado;
    char codigo[10];
};

typedef struct {
    NoMedico* caminho[MAX_CAMINHO];
    float probabilidade_total;
    int profundidade;
    int tamanho_caminho;
} Diagnostico;

typedef struct {
    Diagnostico diagnosticos[MAX_DIAGNOSTICOS];
    int num_diagnosticos;
    int profundidade_limite;
    int nos_explorados;
    int diagnosticos_truncados;
} ResultadoDiagnostico;

//cria um nó com nome, tipo e código
static NoMedico* criarNoMedico(const char* nome, TipoNo tipo, const char* codigo) {
    NoMedico* no = (NoMedico*)malloc(sizeof(NoMedico));
    strncpy(no->nome, nome, MAX_NOME-1);
    no->nome[MAX_NOME-1] = '\0';
    no->tipo = tipo;
    strncpy(no->codigo, codigo, 9);
    no->codigo[9] = '\0';
    no->num_conexoes = 0;
    no->visitado = 0;
    return no;
}

//conecta dois nós com probabilidade e descrição
static void adicionarConexao(NoMedico* origem, NoMedico* destino, float probabilidade, const char* descricao) {
    if(origem->num_conexoes >= MAX_CONEXOES) {
        printf("Erro: conexoes demais em %s\n", origem->nome);
        return;
    }
    Conexao* con = &origem->conexoes[origem->num_conexoes++];
    con->destino = destino;
    con->probabilidade = probabilidade;
    strncpy(con->descricao, descricao, MAX_NOME-1);
    con->descricao[MAX_NOME-1] = '\0';
}

//busca recursiva até a profundidade máxima
static void buscarDiagnostico(NoMedico* atual, int prof_max, int prof_atual, NoMedico* caminho[], float prob_acum, ResultadoDiagnostico* res) {
    if(prof_atual >= prof_max) {
        if(atual->tipo != DOENCA) res->diagnosticos_truncados++;
        return;
    }

    atual->visitado = 1;
    res->nos_explorados++;
    caminho[prof_atual] = atual;

    if(atual->tipo == DOENCA) {
        if(res->num_diagnosticos < MAX_DIAGNOSTICOS) {
            Diagnostico* d = &res->diagnosticos[res->num_diagnosticos++];
            for(int i = 0; i <= prof_atual; i++) d->caminho[i] = caminho[i];
            d->tamanho_caminho = prof_atual + 1;
            d->probabilidade_total = prob_acum;
            d->profundidade = prof_atual;
            printf("Diagnostico encontrado: %s (prob=%.2f)\n", atual->nome, prob_acum);
        }
        atual->visitado = 0;
        return;
    }

    for(int i = 0; i < atual->num_conexoes; i++) {
        Conexao* con = &atual->conexoes[i];
        NoMedico* prox = con->destino;
        if(prox->visitado) continue;

        printf("Seguindo: %s (%.2f) -> %s\n", con->descricao, con->probabilidade, prox->nome);
        float nova_prob = prob_acum * con->probabilidade;
        buscarDiagnostico(prox, prof_max, prof_atual + 1, caminho, nova_prob, res);
    }

    atual->visitado = 0;
}

//ordena os diagnósticos do mais provável pro menos provável
static void ordenarDiagnosticos(ResultadoDiagnostico* res) {
    for(int i = 0; i < res->num_diagnosticos - 1; i++) {
        for(int j = 0; j < res->num_diagnosticos - i - 1; j++) {
            if(res->diagnosticos[j].probabilidade_total < res->diagnosticos[j+1].probabilidade_total) {
                Diagnostico tmp = res->diagnosticos[j];
                res->diagnosticos[j] = res->diagnosticos[j+1];
                res->diagnosticos[j+1] = tmp;
            }
        }
    }
}

//mostra o resultado final da busca
static void imprimirResultadoDiagnostico(ResultadoDiagnostico* res, const char* sintoma_inicial) {
    printf("\nRELATORIO DE DIAGNOSTICO MEDICO\n");
    printf("================================\n");
    printf("Sintoma inicial: %s\n", sintoma_inicial);
    printf("Profundidade maxima: %d\n", res->profundidade_limite);
    printf("Nos explorados: %d\n", res->nos_explorados);
    printf("Diagnosticos encontrados: %d\n", res->num_diagnosticos);
    printf("Truncados (chegou no limite sem achar doenca): %d\n", res->diagnosticos_truncados);

    if(res->num_diagnosticos == 0) {
        printf("\nNenhum diagnostico encontrado dentro do limite.\n");
        if(res->diagnosticos_truncados > 0) {
            printf("Tenta aumentar a profundidade e ver no que dá.\n");
        }
        return;
    }

    ordenarDiagnosticos(res);

    printf("\nDIAGNOSTICOS (do mais pro menos provável):\n");
    for(int i = 0; i < res->num_diagnosticos; i++) {
        Diagnostico* d = &res->diagnosticos[i];
        NoMedico* alvo = d->caminho[d->tamanho_caminho-1];
        printf("%d) %s (CID=%s) prob=%.1f%% prof=%d\n", i+1, alvo->nome, alvo->codigo, d->probabilidade_total*100.0f, d->profundidade);
        printf("Cadeia: [");
        for(int j = 0; j < d->tamanho_caminho; j++) {
            printf("%s", d->caminho[j]->nome);
            if(j < d->tamanho_caminho-1) printf(", ");
        }
        printf("]\n\n");
    }

    float p0 = res->diagnosticos[0].probabilidade_total;
    printf("Recomendacoes:\n");
    if(p0 >= 0.7f) {
        printf("- Alta probabilidade, vale investigar com exames mesmo.\n");
    } else if(p0 >= 0.4f) {
        printf("- Pode ser, melhor investigar e comparar com outros sintomas.\n");
    } else {
        printf("- Baixa chance, talvez precise olhar outros sinais ou aumentar a profundidade.\n");
    }
    if(res->diagnosticos_truncados > 0) {
        printf("- Tem caminhos que pararam antes da hora, talvez escondam algo.\n");
    }
}

//monta a base com os sintomas, causas e doenças
//isso aqui é o "grafo" que a busca vai percorrer
//base criada manualmente pra simular um grafo de conhecimento médico simples
static NoMedico** criarBaseConhecimentoMedico(int* num_nos) {
    *num_nos = 20;
    NoMedico** nos = (NoMedico**)malloc((*num_nos) * sizeof(NoMedico*));

    //os sintomas
    nos[0] = criarNoMedico("Febre alta", SINTOMA, "S01");
    nos[1] = criarNoMedico("Dor de cabeca intensa", SINTOMA, "S02");
    nos[2] = criarNoMedico("Dor abdominal", SINTOMA, "S03");
    nos[3] = criarNoMedico("Tosse persistente", SINTOMA, "S04");

    //causas intermediárias
    nos[4] = criarNoMedico("Infeccao bacteriana", CAUSA_INTERMEDIARIA, "C01");
    nos[5] = criarNoMedico("Infeccao viral", CAUSA_INTERMEDIARIA, "C02");
    nos[6] = criarNoMedico("Inflamacao", CAUSA_INTERMEDIARIA, "C03");
    nos[7] = criarNoMedico("Obstrucao intestinal", CAUSA_INTERMEDIARIA, "C04");
    nos[8] = criarNoMedico("Processo alergico", CAUSA_INTERMEDIARIA, "C05");
    nos[9] = criarNoMedico("Disturbio neurologico", CAUSA_INTERMEDIARIA, "C06");
    nos[10] = criarNoMedico("Infeccao respiratoria", CAUSA_INTERMEDIARIA, "C07");

    //as doenças mesmo
    nos[11] = criarNoMedico("Pneumonia bacteriana", DOENCA, "J15.9");
    nos[12] = criarNoMedico("Gripe viral", DOENCA, "J11.1");
    nos[13] = criarNoMedico("Meningite", DOENCA, "G03.9");
    nos[14] = criarNoMedico("Apendicite", DOENCA, "K37");
    nos[15] = criarNoMedico("Gastroenterite", DOENCA, "K59.1");
    nos[16] = criarNoMedico("Enxaqueca", DOENCA, "G43.9");
    nos[17] = criarNoMedico("Asma", DOENCA, "J45.9");
    nos[18] = criarNoMedico("Bronquite", DOENCA, "J40");
    nos[19] = criarNoMedico("Sinusite", DOENCA, "J32.9");

    //conexões entre sintomas/causas/doenças
    adicionarConexao(nos[0], nos[4], 0.7f, "indica possivel infeccao bacteriana");
    adicionarConexao(nos[0], nos[5], 0.6f, "pode ser causada por virus");
    adicionarConexao(nos[0], nos[3], 0.4f, "pode indicar inflamacao");
    adicionarConexao(nos[1], nos[6], 0.5f, "sugere processo inflamatorio");
    adicionarConexao(nos[1], nos[9], 0.6f, "pode indicar problema neurologico");
    adicionarConexao(nos[1], nos[5], 0.4f, "comum em infeccoes virais");
    adicionarConexao(nos[2], nos[7], 0.8f, "forte indicacao de obstrucao");
    adicionarConexao(nos[2], nos[3], 0.6f, "pode ser processo inflamatorio");
    adicionarConexao(nos[2], nos[4], 0.5f, "possivel infeccao bacteriana");
    adicionarConexao(nos[3], nos[10], 0.8f, "sugere infeccao respiratoria");
    adicionarConexao(nos[3], nos[8], 0.4f, "pode ser alergia");
    adicionarConexao(nos[3], nos[5], 0.6f, "comum em viroses");
    adicionarConexao(nos[4], nos[11], 0.7f, "pneumonia e comum");
    adicionarConexao(nos[4], nos[13], 0.3f, "meningite e possivel");
    adicionarConexao(nos[4], nos[14], 0.8f, "apendicite frequente");
    adicionarConexao(nos[5], nos[12], 0.8f, "gripe e muito comum");
    adicionarConexao(nos[5], nos[15], 0.5f, "gastroenterite viral");
    adicionarConexao(nos[6], nos[16], 0.6f, "enxaqueca por inflamacao");
    adicionarConexao(nos[6], nos[19], 0.5f, "sinusite inflamatoria");
    adicionarConexao(nos[7], nos[14], 0.9f, "apendicite causa obstrucao");
    adicionarConexao(nos[8], nos[17], 0.8f, "asma e reacao alergica");
    adicionarConexao(nos[9], nos[13], 0.7f, "meningite e neurologica");
    adicionarConexao(nos[9], nos[16], 0.8f, "enxaqueca e neurologica");
    adicionarConexao(nos[10], nos[11], 0.6f, "pneumonia");
    adicionarConexao(nos[10], nos[18], 0.7f, "bronquite");
    adicionarConexao(nos[10], nos[19], 0.4f, "sinusite");

    return nos;
}

//compara o que muda ao alterar a profundidade da busca
static void executarDiagnosticoComparativo(NoMedico* sintoma_inicial) {
    int profundidades[] = {1,2,3,4};
    printf("\nANALISE COMPARATIVA (profundidades diferentes)\n");
    for(int i = 0; i < 4; i++) {
        ResultadoDiagnostico res = {0};
        NoMedico* caminho[MAX_CAMINHO];
        res.profundidade_limite = profundidades[i];
        printf("\nProfundidade: %d\n", profundidades[i]);
        buscarDiagnostico(sintoma_inicial, profundidades[i], 0, caminho, 1.0f, &res);
        printf("Diagnosticos: %d | Truncados: %d\n", res.num_diagnosticos, res.diagnosticos_truncados);
        if(res.num_diagnosticos > 0) {
            ordenarDiagnosticos(&res);
            NoMedico* alvo = res.diagnosticos[0].caminho[res.diagnosticos[0].tamanho_caminho-1];
            printf("Melhor: %s (%.1f%%)\n", alvo->nome, res.diagnosticos[0].probabilidade_total*100.0f);
        }
    }
}

int main(void) {
    printf("Diagnostico com profundidade limitada\n\n");

    int num_nos = 0;
    NoMedico** nos = criarBaseConhecimentoMedico(&num_nos);

    NoMedico* sintoma_inicial = nos[0]; //febre alta
    int prof_max = 3;

    ResultadoDiagnostico res = {0};
    res.profundidade_limite = prof_max;
    NoMedico* caminho[MAX_CAMINHO];

    printf("Sintoma: %s\nProfundidade maxima: %d\n\n", sintoma_inicial->nome, prof_max);
    buscarDiagnostico(sintoma_inicial, prof_max, 0, caminho, 1.0f, &res);

    imprimirResultadoDiagnostico(&res, sintoma_inicial->nome);
    executarDiagnosticoComparativo(sintoma_inicial);

    for(int i = 0; i < num_nos; i++) free(nos[i]);
    free(nos);
    return 0;
}