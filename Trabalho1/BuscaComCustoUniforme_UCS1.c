// Feito para a disciplina de Inteligência Artificial
// Simula a busca de custo uniforme (UCS) em um mapa de entregas de drone
// O objetivo é encontrar a rota com menor custo energético entre dois pontos

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#define MAX_VERTICES 100
#define INFINITO FLT_MAX

typedef struct {
    int id;
    float custo_total;
    int pai;
} Nodo;

typedef struct Aresta {
    int destino;
    float custo;
    struct Aresta* proxima;
} Aresta;

typedef struct {
    int num_vertices;
    Aresta** lista_adjacencia;
    char nomes[MAX_VERTICES][50];
} Grafo;

typedef struct {
    Nodo* dados;
    int tamanho;
    int capacidade;
} MinHeap;

//cria uma heap mínima
static MinHeap* criar_min_heap(int capacidade){
    MinHeap* h = (MinHeap*)malloc(sizeof(MinHeap));
    h->dados = (Nodo*)malloc(capacidade * sizeof(Nodo));
    h->tamanho = 0;
    h->capacidade = capacidade;
    return h;
}

//troca dois nodos de lugar
static void trocar_nodo(Nodo* a, Nodo* b){
    Nodo t = *a; *a = *b; *b = t;
}

//sobe um nodo na heap
static void heapify_up(MinHeap* h, int i){
    if(i == 0) return;
    int p = (i - 1) / 2;
    if(h->dados[i].custo_total < h->dados[p].custo_total){
        trocar_nodo(&h->dados[i], &h->dados[p]);
        heapify_up(h, p);
    }
}

//desce um nodo na heap
static void heapify_down(MinHeap* h, int i){
    int m = i, e = 2*i + 1, d = 2*i + 2;
    if(e < h->tamanho && h->dados[e].custo_total < h->dados[m].custo_total) m = e;
    if(d < h->tamanho && h->dados[d].custo_total < h->dados[m].custo_total) m = d;
    if(m != i){
        trocar_nodo(&h->dados[i], &h->dados[m]);
        heapify_down(h, m);
    }
}

//adiciona na heap
static void inserir_heap(MinHeap* h, Nodo n){
    if(h->tamanho >= h->capacidade){
        printf("Erro: heap cheia\n");
        return;
    }
    h->dados[h->tamanho] = n;
    heapify_up(h, h->tamanho);
    h->tamanho++;
}

//tira o menor elemento
static Nodo extrair_minimo(MinHeap* h){
    Nodo invalido = {-1, INFINITO, -1};
    if(h->tamanho == 0){
        printf("Erro: heap vazia\n");
        return invalido;
    }
    Nodo min = h->dados[0];
    h->dados[0] = h->dados[h->tamanho - 1];
    h->tamanho--;
    heapify_down(h, 0);
    return min;
}

static int heap_vazio(const MinHeap* h){
    return h->tamanho == 0;
}

static void liberar_heap(MinHeap* h){
    free(h->dados);
    free(h);
}

//cria o grafo com n vértices
static Grafo* criar_grafo(int n){
    Grafo* g = (Grafo*)malloc(sizeof(Grafo));
    g->num_vertices = n;
    g->lista_adjacencia = (Aresta**)malloc(n * sizeof(Aresta*));
    for(int i = 0; i < n; i++){
        g->lista_adjacencia[i] = NULL;
        sprintf(g->nomes[i], "Local_%d", i);
    }
    return g;
}

//define nome do ponto no grafo
static void definir_nome_local(Grafo* g, int id, const char* nome){
    if(id >= 0 && id < g->num_vertices){
        strncpy(g->nomes[id], nome, 49);
        g->nomes[id][49] = '\0';
    }
}

//cria conexão entre dois pontos
static void adicionar_aresta(Grafo* g, int origem, int destino, float custo){
    Aresta* e = (Aresta*)malloc(sizeof(Aresta));
    e->destino = destino;
    e->custo = custo;
    e->proxima = g->lista_adjacencia[origem];
    g->lista_adjacencia[origem] = e;
}

//mostra o grafo com nomes e custos
static void imprimir_grafo(const Grafo* g){
    printf("Mapa de locais:\n");
    for(int i = 0; i < g->num_vertices; i++){
        printf("De %s: ", g->nomes[i]);
        Aresta* a = g->lista_adjacencia[i];
        if(!a){ printf("sem conexões\n"); continue; }
        while(a){
            printf("-> %s (%.2f)", g->nomes[a->destino], a->custo);
            a = a->proxima;
            if(a) printf(", ");
        }
        printf("\n");
    }
    printf("\n");
}

//mostra o caminho encontrado
static void imprimir_caminho(const Grafo* g, const int* pais, int origem, int destino, float custo_total){
    if(pais[destino] == -1 && origem != destino){
        printf("Caminho nao encontrado.\n");
        return;
    }

    int cont = 0, cur = destino;
    while(cur != -1){ cont++; cur = pais[cur]; }

    int* caminho = (int*)malloc(cont * sizeof(int));
    cur = destino;
    for(int i = cont - 1; i >= 0; i--){
        caminho[i] = cur;
        cur = pais[cur];
    }

    printf("ROTA OTIMA ENCONTRADA\n");
    printf("Custo total: %.2f\n", custo_total);
    printf("Caminho:\n");
    for(int i = 0; i < cont; i++){
        printf("  %d. %s", i+1, g->nomes[caminho[i]]);
        if(i < cont - 1){
            Aresta* a = g->lista_adjacencia[caminho[i]];
            while(a && a->destino != caminho[i+1]) a = a->proxima;
            if(a) printf(" -> (%.2f)", a->custo);
        }
        printf("\n");
    }

    free(caminho);
}

//UCS mesmo, a busca em si
static float busca_custo_uniforme(const Grafo* g, int origem, int destino){
    float* custo = (float*)malloc(g->num_vertices * sizeof(float));
    int* pais = (int*)malloc(g->num_vertices * sizeof(int));
    int* visitado = (int*)malloc(g->num_vertices * sizeof(int));

    for(int i = 0; i < g->num_vertices; i++){
        custo[i] = INFINITO;
        pais[i] = -1;
        visitado[i] = 0;
    }
    custo[origem] = 0.0f;

    MinHeap* heap = criar_min_heap(g->num_vertices * g->num_vertices);
    inserir_heap(heap, (Nodo){origem, 0.0f, -1});

    printf("Iniciando UCS\n");
    printf("Origem: %s\n", g->nomes[origem]);
    printf("Destino: %s\n\n", g->nomes[destino]);

    while(!heap_vazio(heap)){
        Nodo u = extrair_minimo(heap);
        if(visitado[u.id]) continue;
        visitado[u.id] = 1;

        printf("Explorando: %s (custo=%.2f)\n", g->nomes[u.id], u.custo_total);

        if(u.id == destino){
            printf("\nChegou no destino.\n\n");
            imprimir_caminho(g, pais, origem, destino, u.custo_total);
            float resultado = u.custo_total;
            free(custo); free(pais); free(visitado); liberar_heap(heap);
            return resultado;
        }

        Aresta* a = g->lista_adjacencia[u.id];
        while(a){
            int v = a->destino;
            float novo = u.custo_total + a->custo;
            if(novo < custo[v]){
                custo[v] = novo;
                pais[v] = u.id;
                if(!visitado[v]){
                    inserir_heap(heap, (Nodo){v, novo, u.id});
                    printf("  -> Inserindo %s na fila (%.2f)\n", g->nomes[v], novo);
                }
            }
            a = a->proxima;
        }
        printf("\n");
    }

    printf("Caminho nao encontrado.\n");
    free(custo); free(pais); free(visitado); liberar_heap(heap);
    return INFINITO;
}

//monta o grafo do exemplo com lugares reais
static Grafo* criar_grafo_exemplo(void){
    Grafo* g = criar_grafo(8);

    definir_nome_local(g, 0, "Base_Drone");
    definir_nome_local(g, 1, "Centro_Cidade");
    definir_nome_local(g, 2, "Bairro_Norte");
    definir_nome_local(g, 3, "Bairro_Sul");
    definir_nome_local(g, 4, "Zona_Industrial");
    definir_nome_local(g, 5, "Aeroporto");
    definir_nome_local(g, 6, "Hospital");
    definir_nome_local(g, 7, "Shopping");

    adicionar_aresta(g, 0, 1, 5.5f);
    adicionar_aresta(g, 0, 2, 8.0f);
    adicionar_aresta(g, 0, 4, 7.2f);

    adicionar_aresta(g, 1, 0, 5.5f);
    adicionar_aresta(g, 1, 2, 6.8f);
    adicionar_aresta(g, 1, 3, 4.5f);
    adicionar_aresta(g, 1, 7, 3.2f);

    adicionar_aresta(g, 2, 0, 8.0f);
    adicionar_aresta(g, 2, 1, 6.8f);
    adicionar_aresta(g, 2, 5, 12.5f);
    adicionar_aresta(g, 2, 6, 9.3f);

    adicionar_aresta(g, 3, 1, 4.5f);
    adicionar_aresta(g, 3, 4, 6.1f);
    adicionar_aresta(g, 3, 7, 5.8f);

    adicionar_aresta(g, 4, 0, 7.2f);
    adicionar_aresta(g, 4, 3, 6.1f);
    adicionar_aresta(g, 4, 5, 8.9f);

    adicionar_aresta(g, 5, 2, 12.5f);
    adicionar_aresta(g, 5, 4, 8.9f);
    adicionar_aresta(g, 5, 6, 7.4f);

    adicionar_aresta(g, 6, 2, 9.3f);
    adicionar_aresta(g, 6, 5, 7.4f);
    adicionar_aresta(g, 6, 7, 4.1f);

    adicionar_aresta(g, 7, 1, 3.2f);
    adicionar_aresta(g, 7, 3, 5.8f);
    adicionar_aresta(g, 7, 6, 4.1f);

    return g;
}

static void liberar_grafo(Grafo* g){
    for(int i = 0; i < g->num_vertices; i++){
        Aresta* a = g->lista_adjacencia[i];
        while(a){
            Aresta* t = a;
            a = a->proxima;
            free(t);
        }
    }
    free(g->lista_adjacencia);
    free(g);
}

int main(void){
    printf("Drone entregador (energia minima)\n\n");

    Grafo* g = criar_grafo_exemplo();
    imprimir_grafo(g);

    int origem = 0;  //Base_Drone
    int destino = 6; //Hospital

    printf("Missao: entrega de medicamentos\n");
    printf("De: %s -> Para: %s\n\n", g->nomes[origem], g->nomes[destino]);

    float custo_min = busca_custo_uniforme(g, origem, destino);
    if(custo_min != INFINITO){
        printf("\nMissao concluida.\n");
        printf("Energia total consumida: %.2f\n", custo_min);
    }else{
        printf("\nFalha: rota nao encontrada.\n");
    }

    liberar_grafo(g);
    return 0;
}