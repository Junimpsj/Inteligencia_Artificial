// Feito para a disciplina de Inteligência Artificial
// Esse código monta um PC daora dentro do orçamento, testando várias combinações possíveis
// A ideia é achar o melhor custo-benefício usando UCS (Busca com Custo Uniforme)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define MAX_NOME 30
#define MAX_COMPONENTES 50
#define TIPOS_COMPONENTES 6
#define CAP_HEAP 10000

typedef enum {
    TCPU = 0,
    TGPU,
    TRAM,
    TSTORAGE,
    TMOTHERBOARD,
    TPSU
} TipoComponente;

static const char* NOMES_TIPO[TIPOS_COMPONENTES] = {
    "CPU", "GPU", "RAM", "Storage", "Motherboard", "PSU"
};

typedef struct {
    char nome[MAX_NOME];
    float custo;
    int desempenho;
    TipoComponente tipo;
} Componente;

typedef struct {
    Componente componentes[TIPOS_COMPONENTES];
    int sel[TIPOS_COMPONENTES];
    float custo_total;
    int desempenho_total;
    int qtd_sel;
    float score;
} ConfigPC;

typedef struct NoUCS {
    ConfigPC cfg;
    int prox_tipo;
    struct NoUCS* pai;
} NoUCS;

typedef struct {
    NoUCS* dados;
    int tamanho;
    int capacidade;
} MinHeapPC;

typedef struct {
    Componente* lista[TIPOS_COMPONENTES];
    int qtd[TIPOS_COMPONENTES];
} Loja;

// heap pro UCS
static MinHeapPC* criar_heap_pc(int cap){
    MinHeapPC* h = malloc(sizeof(MinHeapPC));
    h->dados = malloc(cap * sizeof(NoUCS));
    h->tamanho = 0;
    h->capacidade = cap;
    return h;
}

static void trocar_no(NoUCS* a, NoUCS* b){
    NoUCS t = *a; *a = *b; *b = t;
}

static void heap_up(MinHeapPC* h, int i){
    if(i == 0) return;
    int p = (i - 1) / 2;
    if(h->dados[i].cfg.custo_total < h->dados[p].cfg.custo_total){
        trocar_no(&h->dados[i], &h->dados[p]);
        heap_up(h, p);
    }
}

static void heap_down(MinHeapPC* h, int i){
    int m = i, e = 2*i + 1, d = 2*i + 2;
    if(e < h->tamanho && h->dados[e].cfg.custo_total < h->dados[m].cfg.custo_total) m = e;
    if(d < h->tamanho && h->dados[d].cfg.custo_total < h->dados[m].cfg.custo_total) m = d;
    if(m != i){
        trocar_no(&h->dados[i], &h->dados[m]);
        heap_down(h, m);
    }
}

static void heap_inserir(MinHeapPC* h, NoUCS n){
    if(h->tamanho >= h->capacidade){
        printf("Heap cheia\n");
        return;
    }
    h->dados[h->tamanho] = n;
    heap_up(h, h->tamanho);
    h->tamanho++;
}

static NoUCS heap_extrair_min(MinHeapPC* h){
    NoUCS vazio; memset(&vazio, 0, sizeof(NoUCS));
    if(h->tamanho == 0){
        printf("Heap vazia\n");
        return vazio;
    }
    NoUCS min = h->dados[0];
    h->dados[0] = h->dados[h->tamanho - 1];
    h->tamanho--;
    heap_down(h, 0);
    return min;
}

static int heap_vazio(const MinHeapPC* h){
    return h->tamanho == 0;
}

static void heap_liberar(MinHeapPC* h){
    free(h->dados);
    free(h);
}

//monta a loja com os componentes disponíveis
static Loja* criar_loja(void){
    Loja* lj = malloc(sizeof(Loja));

    lj->qtd[TCPU] = 5; lj->qtd[TGPU] = 4;
    lj->qtd[TRAM] = 4; lj->qtd[TSTORAGE] = 3;
    lj->qtd[TMOTHERBOARD] = 3; lj->qtd[TPSU] = 3;

    for(int t = 0; t < TIPOS_COMPONENTES; t++)
        lj->lista[t] = malloc(lj->qtd[t] * sizeof(Componente));

        // preencher CPUs
strcpy(lj->lista[TCPU][0].nome, "Ryzen 5 5600G"); lj->lista[TCPU][0].custo = 850; lj->lista[TCPU][0].desempenho = 80; lj->lista[TCPU][0].tipo = TCPU;
strcpy(lj->lista[TCPU][1].nome, "Intel i5 10400F"); lj->lista[TCPU][1].custo = 750; lj->lista[TCPU][1].desempenho = 75; lj->lista[TCPU][1].tipo = TCPU;
strcpy(lj->lista[TCPU][2].nome, "Ryzen 7 5800X"); lj->lista[TCPU][2].custo = 1300; lj->lista[TCPU][2].desempenho = 95; lj->lista[TCPU][2].tipo = TCPU;
strcpy(lj->lista[TCPU][3].nome, "Intel i7 11700K"); lj->lista[TCPU][3].custo = 1400; lj->lista[TCPU][3].desempenho = 92; lj->lista[TCPU][3].tipo = TCPU;
strcpy(lj->lista[TCPU][4].nome, "Athlon 3000G"); lj->lista[TCPU][4].custo = 350; lj->lista[TCPU][4].desempenho = 40; lj->lista[TCPU][4].tipo = TCPU;

// preencher GPUs
strcpy(lj->lista[TGPU][0].nome, "RTX 3060"); lj->lista[TGPU][0].custo = 1900; lj->lista[TGPU][0].desempenho = 90; lj->lista[TGPU][0].tipo = TGPU;
strcpy(lj->lista[TGPU][1].nome, "GTX 1660 Super"); lj->lista[TGPU][1].custo = 1200; lj->lista[TGPU][1].desempenho = 70; lj->lista[TGPU][1].tipo = TGPU;
strcpy(lj->lista[TGPU][2].nome, "RX 6600"); lj->lista[TGPU][2].custo = 1500; lj->lista[TGPU][2].desempenho = 80; lj->lista[TGPU][2].tipo = TGPU;
strcpy(lj->lista[TGPU][3].nome, "GT 1030"); lj->lista[TGPU][3].custo = 500; lj->lista[TGPU][3].desempenho = 30; lj->lista[TGPU][3].tipo = TGPU;

// preencher RAMs
strcpy(lj->lista[TRAM][0].nome, "8GB DDR4 2666MHz"); lj->lista[TRAM][0].custo = 150; lj->lista[TRAM][0].desempenho = 40; lj->lista[TRAM][0].tipo = TRAM;
strcpy(lj->lista[TRAM][1].nome, "16GB DDR4 3200MHz"); lj->lista[TRAM][1].custo = 300; lj->lista[TRAM][1].desempenho = 75; lj->lista[TRAM][1].tipo = TRAM;
strcpy(lj->lista[TRAM][2].nome, "32GB DDR4 3600MHz"); lj->lista[TRAM][2].custo = 550; lj->lista[TRAM][2].desempenho = 90; lj->lista[TRAM][2].tipo = TRAM;
strcpy(lj->lista[TRAM][3].nome, "4GB DDR4 2133MHz"); lj->lista[TRAM][3].custo = 80; lj->lista[TRAM][3].desempenho = 20; lj->lista[TRAM][3].tipo = TRAM;

// preencher storages
strcpy(lj->lista[TSTORAGE][0].nome, "SSD 480GB"); lj->lista[TSTORAGE][0].custo = 200; lj->lista[TSTORAGE][0].desempenho = 80; lj->lista[TSTORAGE][0].tipo = TSTORAGE;
strcpy(lj->lista[TSTORAGE][1].nome, "HD 1TB 7200RPM"); lj->lista[TSTORAGE][1].custo = 250; lj->lista[TSTORAGE][1].desempenho = 50; lj->lista[TSTORAGE][1].tipo = TSTORAGE;
strcpy(lj->lista[TSTORAGE][2].nome, "SSD NVMe 1TB"); lj->lista[TSTORAGE][2].custo = 600; lj->lista[TSTORAGE][2].desempenho = 95; lj->lista[TSTORAGE][2].tipo = TSTORAGE;

// preencher motherboards
strcpy(lj->lista[TMOTHERBOARD][0].nome, "B450M Gigabyte"); lj->lista[TMOTHERBOARD][0].custo = 450; lj->lista[TMOTHERBOARD][0].desempenho = 60; lj->lista[TMOTHERBOARD][0].tipo = TMOTHERBOARD;
strcpy(lj->lista[TMOTHERBOARD][1].nome, "B660M Asus"); lj->lista[TMOTHERBOARD][1].custo = 700; lj->lista[TMOTHERBOARD][1].desempenho = 80; lj->lista[TMOTHERBOARD][1].tipo = TMOTHERBOARD;
strcpy(lj->lista[TMOTHERBOARD][2].nome, "A320M Biostar"); lj->lista[TMOTHERBOARD][2].custo = 300; lj->lista[TMOTHERBOARD][2].desempenho = 40; lj->lista[TMOTHERBOARD][2].tipo = TMOTHERBOARD;

// preencher PSUs
strcpy(lj->lista[TPSU][0].nome, "Fonte 500W"); lj->lista[TPSU][0].custo = 250; lj->lista[TPSU][0].desempenho = 60; lj->lista[TPSU][0].tipo = TPSU;
strcpy(lj->lista[TPSU][1].nome, "Fonte 600W 80+ Bronze"); lj->lista[TPSU][1].custo = 400; lj->lista[TPSU][1].desempenho = 85; lj->lista[TPSU][1].tipo = TPSU;
strcpy(lj->lista[TPSU][2].nome, "Fonte 400W"); lj->lista[TPSU][2].custo = 180; lj->lista[TPSU][2].desempenho = 40; lj->lista[TPSU][2].tipo = TPSU;


    return lj;
}

static void liberar_loja(Loja* lj){
    for(int t = 0; t < TIPOS_COMPONENTES; t++)
        free(lj->lista[t]);
    free(lj);
}

//calcula desempenho total (ponderado)
static int calcular_desempenho_total(const ConfigPC* c){
    if(c->qtd_sel != TIPOS_COMPONENTES) return 0;
    float wcpu=0.3f, wgpu=0.35f, wram=0.15f;
    float wsto=0.1f, wmb=0.05f, wpsu=0.05f;
    float s = c->componentes[TCPU].desempenho*wcpu +
              c->componentes[TGPU].desempenho*wgpu +
              c->componentes[TRAM].desempenho*wram +
              c->componentes[TSTORAGE].desempenho*wsto +
              c->componentes[TMOTHERBOARD].desempenho*wmb +
              c->componentes[TPSU].desempenho*wpsu;
    return (int)s;
}

//mostra todos os componentes da loja
static void imprimir_loja(const Loja* lj){
    printf("LOJA DE COMPONENTES\n");
    for(int t = 0; t < TIPOS_COMPONENTES; t++){
        printf("%s:\n", NOMES_TIPO[t]);
        for(int i = 0; i < lj->qtd[t]; i++){
            const Componente* c = &lj->lista[t][i];
            printf("  %2d. %-20s  R$ %7.2f  Perf %2d\n", i+1, c->nome, c->custo, c->desempenho);
        }
        printf("\n");
    }
}

//mostra uma configuração completa
static void imprimir_config(const ConfigPC* c, const char* titulo){
    printf("\n%s\n", titulo);
    for(int i = 0; i < (int)strlen(titulo); i++) printf("=");
    printf("\n");
    for(int t = 0; t < TIPOS_COMPONENTES; t++){
        if(c->sel[t]){
            printf("%-12s: %-20s (R$ %7.2f | Perf %2d)\n",
                   NOMES_TIPO[t], c->componentes[t].nome,
                   c->componentes[t].custo, c->componentes[t].desempenho);
        }
    }
    printf("\nCusto total: R$ %.2f\n", c->custo_total);
    printf("Desempenho: %d\n", c->desempenho_total);
    printf("Custo-beneficio: %.4f\n", c->score);
}

//UCS adaptado pra montar PCs
static ConfigPC ucs_pc(const Loja* lj, float orcamento){
    MinHeapPC* heap = criar_heap_pc(CAP_HEAP);
    ConfigPC melhor = {0}; // vai guardar a melhor config que acharmos

    NoUCS ini; memset(&ini, 0, sizeof(ini));
    ini.prox_tipo = 0;
    heap_inserir(heap, ini);

    int nos_explorados = 0, configs_avaliadas = 0;

    printf("Iniciando UCS\nOrcamento maximo: R$ %.2f\n\n", orcamento);

    while(!heap_vazio(heap)){
        NoUCS u = heap_extrair_min(heap);
        nos_explorados++;

        if(u.prox_tipo >= TIPOS_COMPONENTES){
            u.cfg.desempenho_total = calcular_desempenho_total(&u.cfg);
            if(u.cfg.custo_total <= orcamento && u.cfg.desempenho_total > 0){
                configs_avaliadas++;
                u.cfg.score = u.cfg.desempenho_total / u.cfg.custo_total;
                if(u.cfg.score > melhor.score){
                    melhor = u.cfg;
                    printf("Nova melhor config (cb=%.4f | custo=R$ %.2f)\n", melhor.score, melhor.custo_total);
                }
            }
            continue;
        }

        int tipo = u.prox_tipo;
        for(int i = 0; i < lj->qtd[tipo]; i++){
            NoUCS v = u;
            v.cfg.componentes[tipo] = lj->lista[tipo][i];
            v.cfg.sel[tipo] = 1;
            v.cfg.custo_total += lj->lista[tipo][i].custo;
            v.cfg.qtd_sel++;
            v.prox_tipo = tipo + 1;

            if(v.cfg.custo_total <= orcamento){
                heap_inserir(heap, v);
            }
        }
    }

    printf("\nNos explorados: %d\nConfigs completas avaliadas: %d\n", nos_explorados, configs_avaliadas);
    heap_liberar(heap);
    return melhor;
}

int main(void){
    printf("Montagem de PC (melhor custo-beneficio)\n\n");

    Loja* loja = criar_loja();
    imprimir_loja(loja);

    float orcamento = 8000.0f;
    printf("Orcamento: R$ %.2f\n\n", orcamento);

    ConfigPC melhor = ucs_pc(loja, orcamento);

    if(melhor.score > 0.0f){
        melhor.desempenho_total = calcular_desempenho_total(&melhor);
        melhor.score = melhor.desempenho_total / (melhor.custo_total > 0 ? melhor.custo_total : 1.0f);
        imprimir_config(&melhor, "MELHOR CONFIGURACAO ENCONTRADA");
        printf("\nConcluido.\n");
    }else{
        printf("Nenhuma configuracao valida encontrada dentro do orcamento.\n");
    }

    liberar_loja(loja);
    return 0;
}