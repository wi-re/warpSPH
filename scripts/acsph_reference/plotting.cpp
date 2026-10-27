// Output-only adapter for the dynGPUSPH reference. No TecIO dependency.
// Binary fields are native little-endian FP64, with an int32 marker array.
#include "plotting.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/stat.h>
#include <vector>
static std::vector<double> referenceMass;
void reference_set_masses(const double* m, int n) { referenceMass.assign(m, m+n); }

void tecplot_header(bool gas) {
    if (gas) { std::fprintf(stderr, "Gas output is unsupported by the reference adapter\n"); std::exit(2); }
    mkdir("snapshots", 0755);
}
void writeDataToFile(vect* x, double* p, double* rho, vect* v, int* bc,
                     int n, double t, vect*, vect*, double* lambda, double* volume, double* aux) {
    static int frame = 0;
    if ((int)referenceMass.size() != n) { std::fprintf(stderr, "Reference adapter requires fixed particle membership\n"); std::exit(2); }
    std::vector<double> physicalVolume(n);
    for (int i=0; i<n; ++i) physicalVolume[i] = referenceMass[i]/rho[i];
    volume = physicalVolume.data();
    char name[128]; std::snprintf(name, sizeof(name), "snapshots/%06d.bin", frame++);
    FILE* f = std::fopen(name, "wb");
    if (!f) { std::perror(name); std::exit(2); }
    auto put = [f](const void* data, size_t bytes) {
        if (std::fwrite(data, 1, bytes, f) != bytes) { std::perror("snapshot"); std::exit(2); }
    };
    put(&n, sizeof(int)); put(&t, sizeof(double));
    put(x, n*sizeof(vect)); put(p, n*sizeof(double)); put(rho, n*sizeof(double));
    put(v, n*sizeof(vect)); put(bc, n*sizeof(int)); put(lambda, n*sizeof(double));
    put(volume, n*sizeof(double)); put(aux, n*sizeof(double));
    std::fclose(f);
    double vmax=0, ke=0, pmin=INFINITY, pmax=-INFINITY;
    int nf=0;
    for (int i=0; i<n; ++i) if (bc[i]==fluidParticle) {
        double speed2 = v[i].x*v[i].x + v[i].y*v[i].y;
        if (!std::isfinite(speed2) || !std::isfinite(p[i]) || !std::isfinite(x[i].x) || !std::isfinite(x[i].y)) {
            std::fprintf(stderr, "REFERENCE NONFINITE frame=%d particle=%d\n", frame-1, i); std::exit(3);
        }
        vmax = std::fmax(vmax, std::sqrt(speed2)); ke += 0.5*rho[i]*volume[i]*speed2;
        pmin = std::fmin(pmin,p[i]); pmax = std::fmax(pmax,p[i]); ++nf;
    }
    std::printf("REFERENCE SNAPSHOT frame=%d t=%.17g fluid=%d vmax=%.9g ke=%.9g p=[%.9g,%.9g]\n", frame-1,t,nf,vmax,ke,pmin,pmax);
    std::fflush(stdout);
}
void close_tecplot_files(bool) {}
