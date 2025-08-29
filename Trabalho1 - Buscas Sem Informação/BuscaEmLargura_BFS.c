// Feito para a disciplina de Inteligência Artificial
// Simula uma busca em largura num prédio com múltiplos andares
// O objetivo é encontrar o menor caminho até a vítima, considerando paredes e portas

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define LIVRE 0
#define PAREDE 1
#define PORTA 2
#define VITIMA 3

#define ANDARES 3
#define LINHAS 4
#define COLUNAS 4
#define LIMITE (ANDARES*LINHAS*COLUNAS)
#define INFINITO 1000000007

typedef struct {
    int a, l, c;
} Pos;

typedef struct {
    Pos dados[LIMITE];
    int ini, fim, tam;
} Fila;

void iniciar_fila(Fila *f) {
    f->ini = 0;
    f->fim = -1;
    f->tam = 0;
}

bool fila_vazia(Fila *f) {
    return f->tam == 0;
}

void empurra_frente(Fila *f, Pos p) {
    if (f->tam == LIMITE) return;
    f->ini = (f->ini - 1 + LIMITE) % LIMITE;
    f->dados[f->ini] = p;
    f->tam++;
    if (f->fim < 0) f->fim = f->ini;
}

void empurra_fim(Fila *f, Pos p) {
    if (f->tam == LIMITE) return;
    f->fim = (f->fim + 1) % LIMITE;
    f->dados[f->fim] = p;
    f->tam++;
    if (f->tam == 1) f->ini = f->fim;
}

Pos tira_frente(Fila *f) {
    Pos p = {-1, -1, -1};
    if (f->tam == 0) return p;
    p = f->dados[f->ini];
    f->ini = (f->ini + 1) % LIMITE;
    f->tam--;
    if (f->tam == 0) f->ini = 0, f->fim = -1;
    return p;
}

bool valido(int a, int l, int c) {
    return a >= 0 && a < ANDARES && l >= 0 && l < LINHAS && c >= 0 && c < COLUNAS;
}

//monta um prédio aleatório com espaço livre, paredes e portas
void monta_predio(int p[ANDARES][LINHAS][COLUNAS]) {
    srand(time(NULL));

    for (int a = 0; a < ANDARES; a++) {
        for (int l = 0; l < LINHAS; l++) {
            for (int c = 0; c < COLUNAS; c++) {
                int r = rand() % 10;
                if (r < 6) p[a][l][c] = LIVRE;
                else if (r < 8) p[a][l][c] = PORTA;
                else p[a][l][c] = PAREDE;  
            }
        }
    }

    //coloca a vítima em uma posipção aleatória (que não seja a parede)
    while (1) {
        int a = rand() % ANDARES;
        int l = rand() % LINHAS;
        int c = rand() % COLUNAS;
        if (p[a][l][c] != PAREDE) {
            p[a][l][c] = VITIMA;
            break;
        }
    }

    //só pra confirmar que a primeira posição vai sempre ser livre
    p[0][0][0] = LIVRE;
}

void mostra_predio(int p[ANDARES][LINHAS][COLUNAS]) {
    printf("Mapa do predio:\n");
    for (int a = 0; a < ANDARES; a++) {
        printf("Andar %d:\n", a);
        for (int l = 0; l < LINHAS; l++) {
            printf("[");
            for (int c = 0; c < COLUNAS; c++) {
                printf("%d%s", p[a][l][c], c == COLUNAS-1 ? "" : ", ");
            }
            printf("]\n");
        }
        printf("\n");
    }
}

int refaz_caminho(Pos fim, int pa[ANDARES][LINHAS][COLUNAS], int pl[ANDARES][LINHAS][COLUNAS], int pc[ANDARES][LINHAS][COLUNAS], Pos cam[LIMITE]) {
    int t = 0;
    Pos atual = fim;

    while (atual.a != -1) {
        cam[t++] = atual;
        int na = pa[atual.a][atual.l][atual.c];
        int nl = pl[atual.a][atual.l][atual.c];
        int nc = pc[atual.a][atual.l][atual.c];
        if (na == -1) break;
        atual = (Pos){na, nl, nc};
    }

    for (int i = 0, j = t - 1; i < j; i++, j--) {
        Pos tmp = cam[i];
        cam[i] = cam[j];
        cam[j] = tmp;
    }

    return t;
}

int bfs(int predio[ANDARES][LINHAS][COLUNAS], Pos inicio, Pos *destino, Pos caminho[LIMITE]) {
    int dist[ANDARES][LINHAS][COLUNAS];
    int pa[ANDARES][LINHAS][COLUNAS], pl[ANDARES][LINHAS][COLUNAS], pc[ANDARES][LINHAS][COLUNAS];

    for (int a = 0; a < ANDARES; a++)
        for (int l = 0; l < LINHAS; l++)
            for (int c = 0; c < COLUNAS; c++)
                dist[a][l][c] = INFINITO, pa[a][l][c] = pl[a][l][c] = pc[a][l][c] = -1;

    int da[6] = {0, 0, 0, 0, -1, 1};
    int dl[6] = {-1, 1, 0, 0, 0, 0};
    int dc[6] = {0, 0, -1, 1, 0, 0};

    Fila f;
    iniciar_fila(&f);
    dist[inicio.a][inicio.l][inicio.c] = 0;
    empurra_frente(&f, inicio);

    while (!fila_vazia(&f)) {
        Pos atual = tira_frente(&f);

        if (predio[atual.a][atual.l][atual.c] == VITIMA) {
            *destino = atual;
            return refaz_caminho(atual, pa, pl, pc, caminho);
        }

        for (int k = 0; k < 6; k++) {
            int na = atual.a + da[k];
            int nl = atual.l + dl[k];
            int nc = atual.c + dc[k];

            if (!valido(na, nl, nc)) continue;
            if (predio[na][nl][nc] == PAREDE) continue;

            int custo = (predio[na][nl][nc] == PORTA);
            int novo = dist[atual.a][atual.l][atual.c] + custo;

            if (novo < dist[na][nl][nc]) {
                dist[na][nl][nc] = novo;
                pa[na][nl][nc] = atual.a;
                pl[na][nl][nc] = atual.l;
                pc[na][nl][nc] = atual.c;

                Pos viz = {na, nl, nc};
                custo == 0 ? empurra_frente(&f, viz) : empurra_fim(&f, viz);
            }
        }
    }

    *destino = (Pos){-1, -1, -1};
    return 0;
}

int main() {
    int predio[ANDARES][LINHAS][COLUNAS];
    monta_predio(predio);
    mostra_predio(predio);

    Pos inicio = {0, 0, 0}, fim, caminho[LIMITE];
    int passos = bfs(predio, inicio, &fim, caminho);

    if (passos && fim.a != -1) {
        int portas = 0;
        for (int i = 1; i < passos; i++)
            if (predio[caminho[i].a][caminho[i].l][caminho[i].c] == PORTA)
                portas++;

        printf("Vitima achada em A:%d L:%d C:%d\n", fim.a, fim.l, fim.c);
        printf("Portas abertas: %d\n", portas);
        printf("Caminho (%d passos):\n", passos);
        for (int i = 0; i < passos; i++) {
            Pos p = caminho[i];
            printf("%d: A=%d L=%d C=%d\n", i, p.a, p.l, p.c);
        }
    } else {
        printf("Nao deu pra achar a vitima.\n");
    }

    return 0;
}