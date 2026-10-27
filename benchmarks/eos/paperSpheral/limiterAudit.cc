// Fixed-pair audit only. Calls the installed Spheral limited-Q implementation;
// no physics overlay, integration or changes to the installed library.
#include "Geometry/Dimension.hh"
#include "ArtificialViscosity/LimitedMonaghanGingoldViscosityView.hh"
#include "NodeList/FluidNodeList.hh"
#include "Material/GammaLawGas.hh"
#include "Material/PhysicalConstants.hh"
#include <algorithm>
#include <cmath>

extern "C" void spheral_limiter_pairs(const double* input,int count,double* output) {
  using namespace Spheral;
  using D=Dim<2>;using V=D::Vector;using T=D::Tensor;using H=D::SymTensor;
  PhysicalConstants units(1.,1.,1.);
  GammaLawGas<D> eos(5./3.,1.,units,-1.e100,1.e100,MaterialPressureMinType::PressureFloor,0.);
  FluidNodeList<D> nodes("static AV audit",eos,2,0,1.e-30,1.e30,.1,input[5]/input[4],100,1.e-100,1.e100);
  Field<D,T> gradients("static gradients",nodes,T::zero());
  FieldList<D,T> g;g.appendField(gradients);
  FieldList<D,double> factors;
  LimitedMonaghanGingoldViscosityView<D> q(2.,1.,false,false,1.,.2);
  q.epsilon2(.01);
  for(int n=0;n<count;++n) {
    const double* a=input+22*n;double* b=output+8*n;
    const V r(a[0],a[1]);const double K=a[4],nh=a[5],nPerh=nh/K;
    nodes.nodesPerSmoothingScale(nPerh);
    gradients[0]=T(a[6],a[7],a[8],a[9]);gradients[1]=T(a[10],a[11],a[12],a[13]);
    const V vi(a[14],a[15]),vj(a[16],a[17]);
    const H Hi(K/a[2],0.,0.,K/a[2]),Hj(K/a[3],0.,0.,K/a[3]);
    const V ei=Hi*r,ej=Hj*r;
    const V half=.5*r;
    const double si=(gradients[0]*half).dot(half),sj=(gradients[1]*half).dot(half);
    const double ri=si/(sgn(sj)*std::max(1.e-30,std::abs(sj)));
    const double rj=sj/(sgn(si)*std::max(1.e-30,std::abs(si)));
    double phi=limiterVL(std::min(ri,rj));
    const double eta=std::min(ei.magnitude(),ej.magnitude()),crit=1./nPerh,fold=.2/nPerh;
    if(eta<crit)phi*=std::exp(-std::pow((eta-crit)/fold,2));
    double pi,pj,qi,qj;
    q.QPiij(pi,pj,qi,qj,0,0,0,1,r,Hi,ei,vi,a[18],a[20],V::zero(),Hj,ej,vj,a[19],a[21],
        factors.view(),factors.view(),g.view());
    b[0]=phi;b[1]=qi;b[2]=qj;b[3]=a[2]/K;b[4]=a[3]/K;b[5]=eta;b[6]=crit;b[7]=fold;
  }
}
