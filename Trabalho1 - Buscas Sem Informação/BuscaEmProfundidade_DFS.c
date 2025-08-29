// Feito para a disciplina de Inteligencia Artificial
// Implementa busca em profundidade para analisar uma árvore genealógica
// Permite encontrar o ancestral mais distante e o caminho até ele

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NOME 50
#define MAX_CAMINHO 64

//struct de pessoa
struct Pessoa {
    char nome[MAX_NOME];
    int ano_nascimento;
    struct Pessoa* pai;
    struct Pessoa* mae;
    int visitado;
};

//criamos uma nova pessoa com nome e ano
static struct Pessoa* criarPessoa(const char* nome, int ano) {
    struct Pessoa* p = (struct Pessoa*)malloc(sizeof(struct Pessoa));
    strncpy(p->nome, nome, MAX_NOME - 1);
    p->nome[MAX_NOME - 1] = '\0';
    p->ano_nascimento = ano;
    p->pai = NULL;
    p->mae = NULL;
    p->visitado = 0;
    return p;
}

//definimos quem são os pais de alguem
static void definirPais(struct Pessoa* filho, struct Pessoa* pai, struct Pessoa* mae) {
    if (filho == NULL) return;
    filho->pai = pai;
    filho->mae = mae;
}

//zera visitas da árvore
static void resetarVisitados(struct Pessoa* p) {
    if (p == NULL) return;
    p->visitado = 0;
    resetarVisitados(p->pai);
    resetarVisitados(p->mae);
}

//aqui sim é a função de busca em profundidade em si
static int dfsAncestralMaisDistante(struct Pessoa* atual, int prof, struct Pessoa** out_ancestral) {
    if (atual == NULL || atual->visitado) return prof - 1;

    atual->visitado = 1;

    int melhor_prof = prof;
    struct Pessoa* melhor_ptr = atual;

    int prof_pai = dfsAncestralMaisDistante(atual->pai, prof + 1, out_ancestral);
    if (prof_pai > melhor_prof) {
        melhor_prof = prof_pai;
        melhor_ptr = *out_ancestral;
    }

    int prof_mae = dfsAncestralMaisDistante(atual->mae, prof + 1, out_ancestral);
    if (prof_mae > melhor_prof) {
        melhor_prof = prof_mae;
        melhor_ptr = *out_ancestral;
    }

    *out_ancestral = melhor_ptr;
    return melhor_prof;
}

//procurando uma pessoa na árvore pelo nome
static struct Pessoa* dfsBuscarPorNome(struct Pessoa* atual, const char* nome) {
    if (atual == NULL || atual->visitado) return NULL;

    atual->visitado = 1;

    if (strcmp(atual->nome, nome) == 0) return atual;

    struct Pessoa* r = dfsBuscarPorNome(atual->pai, nome);
    if (r != NULL) return r;

    return dfsBuscarPorNome(atual->mae, nome);
}

//encontrando o caminho de origem até um ancestral específico
static int dfsCaminhoAncestral(struct Pessoa* origem, struct Pessoa* destino, struct Pessoa* caminho[], int pos) {
    if (origem == NULL || origem->visitado) return 0;

    caminho[pos] = origem;
    if (origem == destino) return pos + 1;

    origem->visitado = 1;

    int t = dfsCaminhoAncestral(origem->pai, destino, caminho, pos + 1);
    if (t > 0) return t;

    t = dfsCaminhoAncestral(origem->mae, destino, caminho, pos + 1);
    if (t > 0) return t;

    origem->visitado = 0; // desfaz marcação se não deu certo
    return 0;
}

//nessa função de impressão, imprimimos com identação por nível
static void imprimirArvore(struct Pessoa* p, int nivel) {
    if (p == NULL) return;
    for (int i = 0; i < nivel; i++) printf("--");
    printf("%s (%d)\n", p->nome, p->ano_nascimento);
    imprimirArvore(p->pai, nivel + 1);
    imprimirArvore(p->mae, nivel + 1);
}

//aqui criei uma função para criar um exemplo de árvore com algumas pessoas ficticias para cadastrar
static struct Pessoa* criarArvoreExemplo(void) {
    struct Pessoa* ana       = criarPessoa("Ana Silva", 2000);
    struct Pessoa* carlos    = criarPessoa("Carlos Silva", 1975);
    struct Pessoa* maria     = criarPessoa("Maria Santos", 1978);
    struct Pessoa* joao      = criarPessoa("Joao Silva", 1950);
    struct Pessoa* helena    = criarPessoa("Helena Costa", 1952);
    struct Pessoa* antonio   = criarPessoa("Antonio Santos", 1948);
    struct Pessoa* rosa      = criarPessoa("Rosa Lima", 1955);
    struct Pessoa* pedro     = criarPessoa("Pedro Silva", 1920);
    struct Pessoa* francisca = criarPessoa("Francisca Oliveira", 1925);
    struct Pessoa* manuel    = criarPessoa("Manuel Costa", 1922);
    struct Pessoa* carmen    = criarPessoa("Carmen Souza", 1930);
    struct Pessoa* jose      = criarPessoa("Jose Santos", 1915);
    struct Pessoa* conceicao = criarPessoa("Conceicao Lima", 1920);
    struct Pessoa* vicente   = criarPessoa("Vicente Silva", 1890);
    struct Pessoa* esperanca = criarPessoa("Esperanca Rocha", 1895);

    definirPais(ana, carlos, maria);
    definirPais(carlos, joao, helena);
    definirPais(maria, antonio, rosa);
    definirPais(joao, pedro, francisca);
    definirPais(helena, manuel, carmen);
    definirPais(antonio, jose, conceicao);
    definirPais(pedro, vicente, esperanca);

    return ana;
}

int main(void) {
    printf("Analise genealogica \n\n");

    struct Pessoa* raiz = criarArvoreExemplo();

    printf("Arvore (origem = %s):\n", raiz->nome);
    imprimirArvore(raiz, 0);

    resetarVisitados(raiz);
    struct Pessoa* anc = NULL;
    int prof_max = dfsAncestralMaisDistante(raiz, 0, &anc);
    printf("\nAncestral mais distante: %s (%d)\n", anc->nome, anc->ano_nascimento);
    printf("Profundidade (geracoes): [%d]\n", prof_max);

    const char* nome_destino = "Vicente Silva";
    resetarVisitados(raiz);
    struct Pessoa* destino = dfsBuscarPorNome(raiz, nome_destino);
    if (destino == NULL) {
        printf("\nPessoa de destino nao encontrada: %s\n", nome_destino);
    } else {
        resetarVisitados(raiz);
        struct Pessoa* caminho[MAX_CAMINHO];
        int tam = dfsCaminhoAncestral(raiz, destino, caminho, 0);

        if (tam == 0) {
            printf("\nNao ha caminho genealogico de %s ate %s\n", raiz->nome, nome_destino);
        } else {
            printf("\nCaminho genealogico de %s ate %s (%d passos):\n", raiz->nome, nome_destino, tam - 1);
            printf("[");
            for (int i = 0; i < tam; i++) {
                printf("%s%s", caminho[i]->nome, (i == tam - 1 ? "" : ", "));
            }
            printf("]\n");
        }
    }

    const char* nome_busca = "Helena Costa";
    resetarVisitados(raiz);
    struct Pessoa* achou = dfsBuscarPorNome(raiz, nome_busca);
    if (achou) {
        printf("\nPessoa encontrada: %s (%d)\n", achou->nome, achou->ano_nascimento);
        if (achou->pai) printf("Pai: %s (%d)\n", achou->pai->nome, achou->pai->ano_nascimento);
        if (achou->mae) printf("Mae: %s (%d)\n", achou->mae->nome, achou->mae->ano_nascimento);
    } else {
        printf("\nPessoa nao encontrada: %s\n", nome_busca);
    }

    printf("\nConcluido.\n");
    return 0;
}