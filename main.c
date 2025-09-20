// nbody_barnes_hut_perf_views_omp.c
// Build (macOS + Homebrew raylib):
//   # fast single-thread build
//   gcc nbody_barnes_hut_perf_views_omp.c -o nbody -O3 -Ofast -ffast-math -march=native -Wall -Wextra -std=c11 \
//     -I/opt/homebrew/include -L/opt/homebrew/lib \
//     -lraylib -framework Cocoa -framework IOKit -framework OpenGL -lm
//
//   # multi-core (install: brew install libomp)
//   gcc nbody_barnes_hut_perf_views_omp.c -o nbody -O3 -Ofast -ffast-math -march=native -Wall -Wextra -std=c11 \
//     -I/opt/homebrew/include -L/opt/homebrew/lib -Xpreprocessor -fopenmp -lomp \
//     -lraylib -framework Cocoa -framework IOKit -framework OpenGL -lm
//
// Run:
//   ./nbody          # default 20000 bodies, collisions ON
//   ./nbody 30000    # try 30k (tune theta with [ and ])
#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------- Types ----------------
typedef struct { Vector2 pos, vel; float mass, radius; } Body;

typedef struct QNode {
    float cx, cy, h, mass;
    Vector2 com;
    int body;
    struct QNode* child[4];
    int hasChild;
} QNode;

typedef struct { QNode* buf; int cap, used; } NodePool;

typedef struct { int *head, *next; int capCells, capNext, W, H; float originX, originY, cell; } Grid;

typedef enum { VIEW_SPEED=0, VIEW_MASS=1, VIEW_ACCEL=2, VIEW_DENSITY=3, VIEW_RANDOM=4, VIEW_COUNT=5 } ViewMode;

// ---------------- Globals ----------------
static float gTheta = 0.7f; // BH opening angle (relative criterion)
static Texture2D gCircleTex; // shared sprite for batching
static Color gLUT[256];      // color gradient (blue->red)

// ---------------- Small helpers ----------------
static inline float frand(float a, float b){ return a + (b - a) * (float)rand()/(float)RAND_MAX; }
static inline Vector2 vadd(Vector2 a, Vector2 b){ return (Vector2){a.x+b.x, a.y+b.y}; }
static inline Vector2 vsub(Vector2 a, Vector2 b){ return (Vector2){a.x-b.x, a.y-b.y}; }
static inline Vector2 vmul(Vector2 a, float s){ return (Vector2){a.x*s, a.y*s}; }
static inline float  vlen2(Vector2 a){ return a.x*a.x + a.y*a.y; }

static void RecomputeRadius(Body* b){ b->radius = 1.6f + 1.1f * cbrtf(fmaxf(b->mass, 0.001f)); }

static inline Color LUT(float x, float xmin, float xmax){
    if (xmax <= xmin) xmax = xmin + 1e-6f;
    float t = (x - xmin) / (xmax - xmin);
    if (t < 0) t = 0; if (t > 1) t = 1;
    int idx = (int)(t * 255.0f);
    return gLUT[idx];
}

static void BuildGradientLUT(void){
    for (int i=0;i<256;i++){
        float t = i/255.0f;
        float hue = 240.0f * (1.0f - t); // blue->red
        gLUT[i] = ColorFromHSV(hue, 0.85f, 1.0f);
    }
}

static Texture2D MakeCircleSprite(int pix){
    Image im = GenImageColor(pix, pix, BLANK);
    ImageDrawCircleV(&im, (Vector2){(float)pix/2.0f, (float)pix/2.0f}, pix/2, WHITE);
    Texture2D tex = LoadTextureFromImage(im);
    UnloadImage(im);
    return tex;
}

// ---------------- Barnes–Hut ----------------
static QNode* NewNode(NodePool* p, float cx, float cy, float h){
    if (p->used >= p->cap) return NULL;
    QNode* n = &p->buf[p->used++];
    n->cx=cx; n->cy=cy; n->h=h; n->mass=0.0f; n->com=(Vector2){0,0}; n->body=-1; n->hasChild=0;
    for (int i=0;i<4;i++) n->child[i]=NULL;
    return n;
}
static inline int Quadrant(const QNode* n, Vector2 p){
    int east = p.x >= n->cx, north = p.y >= n->cy;
    if (!east && !north) return 0; if ( east && !north) return 1;
    if (!east &&  north) return 2; return 3;
}
static void Subdivide(NodePool* pool, QNode* n){
    float h2 = n->h*0.5f;
    n->child[0] = NewNode(pool, n->cx - h2, n->cy - h2, h2);
    n->child[1] = NewNode(pool, n->cx + h2, n->cy - h2, h2);
    n->child[2] = NewNode(pool, n->cx - h2, n->cy + h2, h2);
    n->child[3] = NewNode(pool, n->cx + h2, n->cy + h2, h2);
    n->hasChild = 1;
}
static inline void Accumulate(QNode* n, float m, Vector2 x){
    float M = n->mass + m;
    if (M > 0) n->com = vmul(vadd(vmul(n->com, n->mass), vmul(x, m)), 1.0f/M);
    n->mass = M;
}
static void InsertBody(NodePool* pool, QNode* n, int bi, Body* bodies){
    Body* b = &bodies[bi];
    if (fabsf(b->pos.x - n->cx) > n->h || fabsf(b->pos.y - n->cy) > n->h) return;
    if (n->body == -1 && !n->hasChild && n->mass == 0.0f){ n->body=bi; n->mass=b->mass; n->com=b->pos; return; }
    if (!n->hasChild){
        Subdivide(pool, n);
        if (n->body != -1){
            int old = n->body; n->body = -1;
            InsertBody(pool, n->child[Quadrant(n, bodies[old].pos)], old, bodies);
        }
    }
    InsertBody(pool, n->child[Quadrant(n, b->pos)], bi, bodies);
    n->mass = 0.0f; n->com = (Vector2){0,0};
    for (int i=0;i<4;i++) if (n->child[i] && n->child[i]->mass>0) Accumulate(n, n->child[i]->mass, n->child[i]->com);
}
static inline int UseCOM(const QNode* n, float s, float r){
    float crit = s / fmaxf(1e-3f, (r - n->h)); // relative criterion
    return (!n->hasChild) || (crit < gTheta);
}
static void BH_Accel(const QNode* n, const Body* bodies, int i, float G, float soft2, Vector2* acc){
    if (!n || n->mass == 0.0f) return;
    if (n->body == i && !n->hasChild) return;
    Vector2 d = vsub(n->com, bodies[i].pos);
    float r2 = d.x*d.x + d.y*d.y + soft2;
    float r  = sqrtf(r2);
    float s  = n->h * 2.0f;
    if (UseCOM(n, s, r)){
        float invR3 = 1.0f / (r2 * r);
        float scale = G * n->mass * invR3;
        acc->x += d.x * scale; acc->y += d.y * scale;
    } else {
        for (int k=0;k<4;k++) BH_Accel(n->child[k], bodies, i, G, soft2, acc);
    }
}
static void DrawNodeRectangles(const QNode* n){
    if (!n || n->mass==0.0f) return;
    DrawRectangleLines((int)(n->cx - n->h), (int)(n->cy - n->h), (int)(n->h*2), (int)(n->h*2), (Color){80,80,120,140});
    if (n->hasChild) for (int i=0;i<4;i++) DrawNodeRectangles(n->child[i]);
}
static void ComputeBounds(const Body* b, int N, float* cx, float* cy, float* half){
    float minx=1e9f, miny=1e9f, maxx=-1e9f, maxy=-1e9f;
    for (int i=0;i<N;i++){ float x=b[i].pos.x, y=b[i].pos.y;
        if (x<minx) minx=x; if (y<miny) miny=y; if (x>maxx) maxx=x; if (y>maxy) maxy=y; }
    *cx = 0.5f*(minx+maxx); *cy = 0.5f*(miny+maxy);
    float halfSpan = fmaxf(maxx-minx, maxy-miny)*0.5f; if (halfSpan<10.0f) halfSpan=10.0f; *half = halfSpan*1.05f;
}

// ---------------- Grid ----------------
static void GridEnsure(Grid* g, int needCells, int needNext){
    if (needCells > g->capCells){ if (g->head) MemFree(g->head); g->head = MemAlloc(sizeof(int)*(size_t)needCells); g->capCells = needCells; }
    if (needNext  > g->capNext ){ if (g->next) MemFree(g->next); g->next = MemAlloc(sizeof(int)*(size_t)needNext ); g->capNext  = needNext; }
}
static void GridBuild(Grid* g, Body* bodies, int N, float minx, float miny, float maxx, float maxy, float cell){
    g->cell=cell; g->originX=minx; g->originY=miny;
    g->W=(int)fmaxf(1.0f, ceilf((maxx - minx)/cell));
    g->H=(int)fmaxf(1.0f, ceilf((maxy - miny)/cell));
    int cells = g->W * g->H;
    GridEnsure(g, cells, N);
    for (int i=0;i<cells;i++) g->head[i] = -1;
    for (int i=0;i<N;i++){
        int cx = (int)floorf((bodies[i].pos.x - g->originX)/cell);
        int cy = (int)floorf((bodies[i].pos.y - g->originY)/cell);
        if (cx<0||cy<0||cx>=g->W||cy>=g->H){ g->next[i] = -1; continue; }
        int idx = cy*g->W + cx;
        g->next[i] = g->head[idx];
        g->head[idx] = i;
    }
}
static inline int GridIndex(const Grid* g, int cx, int cy){
    if (cx<0||cy<0||cx>=g->W||cy>=g->H) return -1; return cy*g->W + cx;
}

// ---------------- Collisions (elastic, duplicate-free) ----------------
static inline void ResolveElastic(Body* a, Body* b, float restitution){
    Vector2 n = vsub(b->pos, a->pos);
    float r2 = n.x*n.x + n.y*n.y; if (r2 <= 0.0f) return;
    float invR = 1.0f/sqrtf(r2);
    float r = r2 * invR; // cheaper than sqrt again
    float rSum = a->radius + b->radius; if (r >= rSum) return;
    Vector2 nhat = (Vector2){ n.x*invR, n.y*invR };
    float invA = 1.0f/a->mass, invB = 1.0f/b->mass, invSum = invA + invB;

    float penetration = rSum - r;
    Vector2 corr = (Vector2){ nhat.x * (penetration/(invSum+1e-6f)), nhat.y * (penetration/(invSum+1e-6f)) };
    a->pos.x -= corr.x * invA; a->pos.y -= corr.y * invA;
    b->pos.x += corr.x * invB; b->pos.y += corr.y * invB;

    Vector2 rv = vsub(b->vel, a->vel);
    float velN = rv.x*nhat.x + rv.y*nhat.y;
    if (velN <= 0.0f){
        float j = -(1.0f + restitution) * velN / (invSum + 1e-6f);
        Vector2 imp = (Vector2){ nhat.x * j, nhat.y * j };
        a->vel.x -= imp.x * invA; a->vel.y -= imp.y * invA;
        b->vel.x += imp.x * invB; b->vel.y += imp.y * invB;
    }
}

// ---------------- Spawn: ring with bands ----------------
static void InitCircleBands(Body* b, int n, float ringR, float jitterR, float massMin, float massMax, float G_for_v){
    float Mtot = 0.0f;
    for (int i=0;i<n;i++){ b[i].mass = frand(massMin, massMax); RecomputeRadius(&b[i]); Mtot += b[i].mass; }
    float v0 = sqrtf(G_for_v * Mtot / fmaxf(ringR, 1.0f));
    const float bands[3] = {0.75f, 1.00f, 1.35f};
    const float probs[3] = {0.25f, 0.5f, 0.25f};

    for (int i=0;i<n;i++){
        float ang = frand(0.0f, 2.0f*(float)M_PI);
        float r   = ringR + frand(-jitterR, jitterR); if (r < 5.0f) r = 5.0f;
        b[i].pos = (Vector2){ r * cosf(ang), r * sinf(ang) };
        float u = frand(0.0f,1.0f);
        int band = (u < probs[0]) ? 0 : (u < probs[0]+probs[1] ? 1 : 2);
        Vector2 t = (Vector2){ -sinf(ang), cosf(ang) };
        float speed = bands[band] * v0 + frand(-0.03f*v0, 0.03f*v0);
        b[i].vel = vmul(t, speed);
    }
    // remove CM drift
    Vector2 P = {0,0}; float mTot = 0.0f;
    for (int i=0;i<n;i++){ P = vadd(P, vmul(b[i].vel, b[i].mass)); mTot += b[i].mass; }
    if (mTot > 0){ Vector2 vcm = (Vector2){ P.x/mTot, P.y/mTot }; for (int i=0;i<n;i++) b[i].vel = vsub(b[i].vel, vcm); }
}

// ---------------- View utils ----------------
static const char* ViewName(ViewMode m){
    switch(m){ case VIEW_SPEED: return "speed"; case VIEW_MASS: return "mass";
        case VIEW_ACCEL: return "accel"; case VIEW_DENSITY: return "density";
        case VIEW_RANDOM: return "random"; default: return "?"; }
}

// ---------------- Main ----------------
int main(int argc, char** argv){
    int N = 20000;
    if (argc >= 2){ int t = atoi(argv[1]); if (t > 0) N = t; }
    if (argc >= 3) srand((unsigned)atoi(argv[2])); else srand((unsigned)time(NULL));

    const int screenW = 1280, screenH = 720;
    InitWindow(screenW, screenH, "Barnes–Hut N-Body (collisions ON, optimized)");
    SetTargetFPS(120);

    BuildGradientLUT();
    gCircleTex = MakeCircleSprite(24); // small white circle; tinted per body

    Body* bodies = MemAlloc(sizeof(Body) * (size_t)N);
    if (!bodies){ CloseWindow(); fprintf(stderr,"Alloc fail\n"); return 1; }

    // Physics params
    const float G = 70.0f;
    const float softening = 14.0f, soft2 = softening*softening;
    const float MASS_MIN = 0.4f, MASS_MAX = 2.5f;

    // Spawn ring
    const float Rring = 1800.0f, Rjitt = 180.0f;
    InitCircleBands(bodies, N, Rring, Rjitt, MASS_MIN, MASS_MAX, G);

    // Camera
    Camera2D cam = {0};
    cam.target = (Vector2){0,0}; cam.offset = (Vector2){screenW*0.5f, screenH*0.5f};
    cam.rotation = 0.0f; cam.zoom = 0.25f;

    // Pools/buffers
    int poolCap = 8*N + 4096;
    QNode*   poolBuf = MemAlloc(sizeof(QNode) * (size_t)poolCap);
    Vector2* acc     = MemAlloc(sizeof(Vector2) * (size_t)N);
    float*   accMag  = MemAlloc(sizeof(float)   * (size_t)N);
    float*   density = MemAlloc(sizeof(float)   * (size_t)N);
    Color*   rndCol  = MemAlloc(sizeof(Color)   * (size_t)N);
    for (int i=0;i<N;i++) rndCol[i] = (Color){ (unsigned char)(30+rand()%200), (unsigned char)(30+rand()%200), (unsigned char)(30+rand()%200), 255 };

    Grid grid = {0};

    // State
    bool paused=false, showTree=false;
    ViewMode view = VIEW_SPEED;

    while (!WindowShouldClose()){
        // Input
        float move = 900.0f * GetFrameTime() / cam.zoom;
        if (IsKeyDown(KEY_LEFT))  cam.target.x -= move;
        if (IsKeyDown(KEY_RIGHT)) cam.target.x += move;
        if (IsKeyDown(KEY_UP))    cam.target.y -= move;
        if (IsKeyDown(KEY_DOWN))  cam.target.y += move;
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyDown(KEY_W)) cam.zoom *= 1.0f + 1.6f*GetFrameTime();
        if (IsKeyDown(KEY_S)) cam.zoom *= 1.0f - 1.6f*GetFrameTime();
        if (cam.zoom < 0.05f) cam.zoom = 0.05f; if (cam.zoom > 6.0f) cam.zoom = 6.0f;
        if (IsKeyPressed(KEY_R)) InitCircleBands(bodies, N, Rring, Rjitt, MASS_MIN, MASS_MAX, G);
        if (IsKeyPressed(KEY_B)) showTree = !showTree;
        if (IsKeyPressed(KEY_V)) view = (ViewMode)((view + 1) % VIEW_COUNT);
        if (IsKeyPressed(KEY_LEFT_BRACKET))  gTheta = fmaxf(0.3f, gTheta - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) gTheta = fminf(1.2f, gTheta + 0.05f);

        // Bounds for grid and culling
        float minx=1e9f, miny=1e9f, maxx=-1e9f, maxy=-1e9f;
        for (int i=0;i<N;i++){ float x=bodies[i].pos.x, y=bodies[i].pos.y;
            if (x<minx) minx=x; if (y<miny) miny=y; if (x>maxx) maxx=x; if (y>maxy) maxy=y; }

        // Build BH tree
        NodePool pool = { .buf = poolBuf, .cap = poolCap, .used = 0 };
        float rootCx, rootCy, rootH;
        ComputeBounds(bodies, N, &rootCx, &rootCy, &rootH);
        QNode* root = NewNode(&pool, rootCx, rootCy, rootH);
        for (int i=0;i<N;i++) InsertBody(&pool, root, i, bodies);

        // Accelerations + adaptive dt (parallelized)
        float amax = 0.0f;
        #pragma omp parallel
        {
            float amax_local = 0.0f;
            #pragma omp for schedule(static)
            for (int i=0;i<N;i++){
                Vector2 a = {0,0};
                BH_Accel(root, bodies, i, G, soft2, &a);
                acc[i] = a;
                float am = sqrtf(a.x*a.x + a.y*a.y);
                accMag[i] = am;
                if (am > amax_local) amax_local = am;
            }
            #pragma omp critical
            { if (amax_local > amax) amax = amax_local; }
        }
        const float dtMax = 1.0f/60.0f, dtMin = 1.0f/4000.0f, eta = 0.25f;
        float dtTarget = (amax > 0.0f) ? (eta * sqrtf(softening / amax)) : dtMax;
        float k = ceilf(log2f(dtMax / fmaxf(dtMin, dtTarget)));
        float dt = dtMax * powf(0.5f, fmaxf(0.0f, k));

        // Integrate (parallelized)
        if (!paused){
            #pragma omp parallel for schedule(static)
            for (int i=0;i<N;i++){ bodies[i].vel.x += acc[i].x * dt; bodies[i].vel.y += acc[i].y * dt; }
            #pragma omp parallel for schedule(static)
            for (int i=0;i<N;i++){ bodies[i].pos.x += bodies[i].vel.x * dt; bodies[i].pos.y += bodies[i].vel.y * dt; }

            // Broadphase grid (cell ~ 2.5x avg radius)
            float avgR = 0.0f; for (int i=0;i<N;i++) avgR += bodies[i].radius; avgR /= (float)N;
            float cell = fmaxf(2.5f*avgR, 8.0f);
            GridBuild(&grid, bodies, N, minx, miny, maxx, maxy, cell);

            // Duplicate-free neighbor set: (0,0) [within cell], (1,0) E, (0,1) N, (1,1) NE, (1,-1) SE
            for (int cy=0; cy<grid.H; cy++){
                for (int cx=0; cx<grid.W; cx++){
                    int idx = GridIndex(&grid, cx, cy);
                    // within cell
                    for (int i = grid.head[idx]; i != -1; i = grid.next[i]){
                        for (int j = grid.next[i]; j != -1; j = grid.next[j]){
                            float rSum = bodies[i].radius + bodies[j].radius;
                            Vector2 d = vsub(bodies[j].pos, bodies[i].pos);
                            if (vlen2(d) <= rSum*rSum) ResolveElastic(&bodies[i], &bodies[j], 0.98f);
                        }
                    }
                    // E
                    int idE = GridIndex(&grid, cx+1, cy);
                    if (idE >= 0){
                        for (int i = grid.head[idx]; i != -1; i = grid.next[i]){
                            for (int j = grid.head[idE]; j != -1; j = grid.next[j]){
                                float rSum = bodies[i].radius + bodies[j].radius;
                                Vector2 d = vsub(bodies[j].pos, bodies[i].pos);
                                if (vlen2(d) <= rSum*rSum) ResolveElastic(&bodies[i], &bodies[j], 0.98f);
                            }
                        }
                    }
                    // N
                    int idN = GridIndex(&grid, cx, cy+1);
                    if (idN >= 0){
                        for (int i = grid.head[idx]; i != -1; i = grid.next[i]){
                            for (int j = grid.head[idN]; j != -1; j = grid.next[j]){
                                float rSum = bodies[i].radius + bodies[j].radius;
                                Vector2 d = vsub(bodies[j].pos, bodies[i].pos);
                                if (vlen2(d) <= rSum*rSum) ResolveElastic(&bodies[i], &bodies[j], 0.98f);
                            }
                        }
                    }
                    // NE
                    int idNE = GridIndex(&grid, cx+1, cy+1);
                    if (idNE >= 0){
                        for (int i = grid.head[idx]; i != -1; i = grid.next[i]){
                            for (int j = grid.head[idNE]; j != -1; j = grid.next[j]){
                                float rSum = bodies[i].radius + bodies[j].radius;
                                Vector2 d = vsub(bodies[j].pos, bodies[i].pos);
                                if (vlen2(d) <= rSum*rSum) ResolveElastic(&bodies[i], &bodies[j], 0.98f);
                            }
                        }
                    }
                    // SE
                    int idSE = GridIndex(&grid, cx+1, cy-1);
                    if (idSE >= 0){
                        for (int i = grid.head[idx]; i != -1; i = grid.next[i]){
                            for (int j = grid.head[idSE]; j != -1; j = grid.next[j]){
                                float rSum = bodies[i].radius + bodies[j].radius;
                                Vector2 d = vsub(bodies[j].pos, bodies[i].pos);
                                if (vlen2(d) <= rSum*rSum) ResolveElastic(&bodies[i], &bodies[j], 0.98f);
                            }
                        }
                    }
                }
            }
        }

        // View metrics
        float vMin=1e30f, vMax=-1e30f;
        if (view == VIEW_SPEED || view == VIEW_RANDOM){
            float smin=1e30f, smax=-1e30f;
            for (int i=0;i<N;i++){ float s2 = bodies[i].vel.x*bodies[i].vel.x + bodies[i].vel.y*bodies[i].vel.y;
                if (s2<smin) smin=s2; if (s2>smax) smax=s2; }
            vMin = sqrtf(smin); vMax = sqrtf(smax);
        } else if (view == VIEW_MASS){
            float mmin=1e30f, mmax=-1e30f;
            for (int i=0;i<N;i++){ float m=bodies[i].mass; if (m<mmin) mmin=m; if (m>mmax) mmax=m; }
            vMin = mmin; vMax = mmax;
        } else if (view == VIEW_ACCEL){
            float amin=1e30f, amax2=-1e30f;
            for (int i=0;i<N;i++){ float a=accMag[i]; if (a<amin) amin=a; if (a>amax2) amax2=a; }
            vMin = amin; vMax = amax2;
        } else if (view == VIEW_DENSITY){
            // light neighbor count grid (reuse grid with tighter cell)
            float cellD = fmaxf(softening*1.5f, 12.0f);
            GridBuild(&grid, bodies, N, minx, miny, maxx, maxy, cellD);
            float dmin=1e30f, dmax=-1e30f;
            for (int i=0;i<N;i++){
                int cx = (int)floorf((bodies[i].pos.x - grid.originX)/grid.cell);
                int cy = (int)floorf((bodies[i].pos.y - grid.originY)/grid.cell);
                int count = 0;
                for (int oy=0; oy<=1; oy++) for (int ox=0; ox<=1; ox++){ // 4 cells are enough for smooth heat
                    int idx = GridIndex(&grid, cx+ox, cy+oy); if (idx<0) continue;
                    for (int j=grid.head[idx]; j!=-1; j=grid.next[j]) count++;
                }
                density[i] = (float)count;
                if (density[i]<dmin) dmin=density[i]; if (density[i]>dmax) dmax=density[i];
            }
            vMin=dmin; vMax=dmax;
        }

        // World rect (for optional culling in draw)
        Vector2 TL = GetScreenToWorld2D((Vector2){0,0}, cam);
        Vector2 BR = GetScreenToWorld2D((Vector2){(float)screenW,(float)screenH}, cam);
        float wx0=fminf(TL.x, BR.x), wy0=fminf(TL.y, BR.y), wx1=fmaxf(TL.x, BR.x), wy1=fmaxf(TL.y, BR.y);
        float pad = 30.0f / cam.zoom;

        BeginDrawing();
        ClearBackground((Color){10,12,18,255});
        BeginMode2D(cam);

        if (showTree) DrawNodeRectangles(root);

        // Batched draw using a single circle texture (fast)
        Rectangle src = {0,0,(float)gCircleTex.width,(float)gCircleTex.height};
        for (int i=0;i<N;i++){
            Vector2 p = bodies[i].pos;
            float r = bodies[i].radius;
            if (p.x < wx0 - pad || p.x > wx1 + pad || p.y < wy0 - pad || p.y > wy1 + pad) continue;

            Color c;
            switch(view){
                case VIEW_SPEED: {
                    float s = sqrtf(bodies[i].vel.x*bodies[i].vel.x + bodies[i].vel.y*bodies[i].vel.y);
                    c = LUT(s, vMin, vMax); break;
                }
                case VIEW_MASS:    c = LUT(bodies[i].mass, vMin, vMax); break;
                case VIEW_ACCEL:   c = LUT(accMag[i], vMin, vMax); break;
                case VIEW_DENSITY: c = LUT(density[i], vMin, vMax); break;
                case VIEW_RANDOM:  default: c = rndCol[i]; break;
            }
            Rectangle dst = { p.x - r, p.y - r, r*2.0f, r*2.0f };
            DrawTexturePro(gCircleTex, src, dst, (Vector2){0,0}, 0.0f, c);
        }

        EndMode2D();

        int fps = GetFPS(); float ms = GetFrameTime()*1000.0f;
        DrawText(TextFormat("Bodies:%d  Zoom:%.2f  Tree nodes:%d  theta:%.2f  FPS:%d  frame:%.2f ms  dt:%.5f  view:%s%s",
                            N, cam.zoom, pool.used, gTheta, fps, ms, dt, ViewName(view),
                            paused ? " [PAUSED]" : ""), 16, 16, 18, RAYWHITE);

        EndDrawing();
    }

    if (grid.head) MemFree(grid.head);
    if (grid.next) MemFree(grid.next);
    UnloadTexture(gCircleTex);
    MemFree(rndCol); MemFree(density); MemFree(accMag); MemFree(acc); MemFree(poolBuf); MemFree(bodies);
    CloseWindow();
    return 0;
}
