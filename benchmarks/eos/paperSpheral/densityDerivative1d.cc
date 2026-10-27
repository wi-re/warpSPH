// Read-only 1D operator audit. No integration or overrides of Spheral symbols.
// Actual native sum volume, RK corrections/gradient and stock sum density;
// paper mass-replacement density and pressure work evaluated with native RK.
#include "Geometry/Dimension.hh"
#include "Kernel/NBSplineKernel.hh"
#include "Kernel/TableKernel.hh"
#include "Material/GammaLawGas.hh"
#include "Material/PhysicalConstants.hh"
#include "NodeList/FluidNodeList.hh"
#include "Neighbor/TreeNeighbor.hh"
#include "DataBase/DataBase.hh"
#include "Hydro/HydroFieldNames.hh"
#include "RK/computeRKSumVolume.hh"
#include "RK/ReproducingKernel.hh"
#include "RK/gradientRK.hh"
#include "CRKSPH/computeCRKSPHSumMassDensity.hh"
#include <memory>
#include <vector>
#include <cmath>

namespace {
using namespace Spheral;
using D=Dim<1>; using Vec=D::Vector;
TableKernel<D>& table(){static NBSplineKernel<D> base(7);static TableKernel<D> W(base,1000);return W;}
}
extern "C" int spheral_density_derivative_version(){return 1;}
// Input rows x,R,m,rho,v,material,real,parent-real-row,u. Real rows precede ghosts.
// Output: V, paper rho, native stock rho, gradientRK(v), continuity(actual rho),
// continuity(paper rho), difference gradient, pressure du, partition gradient,
// A, coefficient[1], volume cap flag, stock density lower-bound flag, pressure ds.
extern "C" int spheral_density_derivative(const double* input,int count,int realCount,double* output){
  PhysicalConstants units(1.,1.,1.);
  GammaLawGas<D> eos(1.4,1.,units,-1.e100,1.e100,MaterialPressureMinType::PressureFloor,0.);
  DataBase<D> db;
  std::vector<std::unique_ptr<FluidNodeList<D>>> nodes;
  std::vector<std::unique_ptr<TreeNeighbor<D>>> neighbors;
  std::vector<std::vector<int>> original;
  std::vector<int> mapK(count),mapJ(count);
  int tags=0;for(int i=0;i<count;++i)tags=std::max(tags,int(input[9*i+5])+1);
  for(int k=0;k<tags;++k){
    std::vector<int> ids;int nreal=0;
    for(int real=1;real>=0;--real)for(int i=0;i<count;++i)
      if(int(input[9*i+5])==k and int(input[9*i+6])==real)ids.push_back(i);
    for(int i:ids)nreal+=int(input[9*i+6]);
    nodes.emplace_back(new FluidNodeList<D>("density derivative "+std::to_string(k),eos,nreal,ids.size()-nreal,
        1.e-8,1.e8,.1,1.,10000,1.e-100,1.e100));
    auto& n=*nodes.back();
    for(size_t j=0;j<ids.size();++j){
      const int id=ids[j];const auto* p=input+9*id;mapK[id]=k;mapJ[id]=j;
      n.positions()[j]=Vec(p[0]);n.Hfield()[j]=D::SymTensor(4./p[1]);
      n.mass()[j]=p[2];n.massDensity()[j]=p[3];n.velocity()[j]=Vec(p[4]);n.specificThermalEnergy()[j]=p[8];
    }
    neighbors.emplace_back(new TreeNeighbor<D>(n,NeighborSearchType::GatherScatter,4.,Vec(-1000.),Vec(1000.)));
    n.registerNeighbor(*neighbors.back());neighbors.back()->updateNodes();db.appendNodeList(n);original.push_back(ids);
  }
  db.updateConnectivityMap(false,false,false);const auto& cm=db.connectivityMap();
  auto x=db.fluidPosition();auto m=db.fluidMass();auto H=db.fluidHfield();auto v=db.fluidVelocity();
  auto V=db.newFluidFieldList(0.,"native geometric volume");
  computeRKSumVolume<D>(cm,table(),x,m,H,V);
  for(int row=realCount;row<count;++row){int parent=int(input[9*row+7]);V(mapK[row],mapJ[row])=V(mapK[parent],mapJ[parent]);}
  ReproducingKernel<D> WR(table(),RKOrder::LinearOrder);
  auto c0=db.newFluidFieldList(RKCoefficients<D>(),"c0");auto c=db.newFluidFieldList(RKCoefficients<D>(),"c1");
  WR.computeCorrections(cm,V,x,H,false,c0,c);
  for(int row=realCount;row<count;++row){int parent=int(input[9*row+7]);c(mapK[row],mapJ[row])=c(mapK[parent],mapJ[parent]);}
  auto gradient=gradientRK<D,Vec>(v,x,V,H,cm,WR,c);
  auto stock=db.newFluidFieldList(0.,"native stock density");computeCRKSPHSumMassDensity<D>(cm,table(),x,m,V,H,stock);
  for(int row=0;row<realCount;++row){
    const int ki=mapK[row],i=mapJ[row];const auto* pi=input+9*row;
    double num=0.,den=0.,diff=0.,partition=0.,pressure=0.;
    const double Vi=V(ki,i),Pi=.4*pi[3]*pi[8];
    for(int other=0;other<count;++other){
      const int kj=mapK[other],j=mapJ[other];const auto* pj=input+9*other;
      const auto rij=x(ki,i)-x(kj,j);const double Vj=V(kj,j);
      const auto w=WR.evaluateKernel(rij,H(kj,j),c(ki,i));
      const auto gi=WR.evaluateGradient(rij,H(kj,j),c(ki,i));
      const auto gj=WR.evaluateGradient(-rij,H(ki,i),c(kj,j));
      const double mij=ki==kj?pj[2]:pi[2];
      num+=mij*Vj*w;den+=Vj*Vj*w;
      diff+=Vj*(pj[4]-pi[4])*gi.x();partition+=Vj*gi.x();
      pressure+=Vi*Vj/pi[2]*(.4*pj[3]*pj[8])*(pi[4]-pj[4])*.5*(gi.x()-gj.x());
    }
    const double rho=num/den,g=gradient(ki,i).xx();auto* out=output+14*row;
    out[0]=Vi;out[1]=rho;out[2]=stock(ki,i);out[3]=g;out[4]=-pi[3]*g;out[5]=-rho*g;
    out[6]=diff;out[7]=pressure;out[8]=partition;out[9]=c(ki,i)[0];out[10]=c(ki,i)[1];
    out[11]=(Vi>=pi[1]*(1.-1.e-12));out[12]=(stock(ki,i)<=.1*pi[2]*4./pi[1]*(1.+1.e-12));
    out[13]=pressure/pi[8]-.4*out[4]/pi[3];
  }
  for(auto& n:nodes)n->unregisterNeighbor();return 0;
}
