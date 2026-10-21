// Static audit only: call the installed SPHSmoothingScale/IdealH operator.
// No integration, overlay, or modification of installed Spheral sources.
#include "Geometry/Dimension.hh"
#include "SmoothingScale/SPHSmoothingScale.hh"
#include "Kernel/NBSplineKernel.hh"
#include "Kernel/TableKernel.hh"
#include "Material/GammaLawGas.hh"
#include "Material/PhysicalConstants.hh"
#include "NodeList/FluidNodeList.hh"
#include "Neighbor/TreeNeighbor.hh"
#include "DataBase/DataBase.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "DataBase/ReplaceBoundedState.hh"
#include "Hydro/HydroFieldNames.hh"
#include <memory>
#include <vector>

namespace {
using namespace Spheral;
using D=Dim<2>;
TableKernel<D>& table() {
  static NBSplineKernel<D> base(7);
  static TableKernel<D> result(base,1000);
  return result;
}
}

extern "C" int spheral_support_table(int which,double* meta,double* data) {
  const auto& W=table();
  if(which==0) {
    const auto& t=W.nPerhInterpolator();
    meta[0]=t.xmin();meta[1]=t.xmax();meta[2]=t.xstep();meta[3]=t.size();
    if(data)for(size_t i=0;i<2*t.size();++i)data[i]=t.data()[i];
    return 2*t.size();
  }
  const auto& t=W.gradWinterpolator();
  meta[0]=t.xmin();meta[1]=t.xmax();meta[2]=t.xstep();meta[3]=t.size();
  if(data)for(size_t i=0;i<t.size();++i)data[i]=t.data()[i];
  return t.size();
}

extern "C" void spheral_support_lookup(const double* x,int n,double* out) {
  for(int i=0;i<n;++i)out[i]=table().equivalentNodesPerSmoothingScale(x[i]);
}

// Rows: x,y,R,m,rho,material,real. Explicit periodic ghosts are supplied by
// the caller; targets are returned for real rows in original input order.
extern "C" int spheral_support_state(const double* input,int count,double nperh,
                                     double damping,double hmin,double hmax,double* output) {
  PhysicalConstants units(1.,1.,1.);
  GammaLawGas<D> eos(5./3.,1.,units,-1.e100,1.e100,MaterialPressureMinType::PressureFloor,0.);
  DataBase<D> db;
  std::vector<std::unique_ptr<FluidNodeList<D>>> nodes;
  std::vector<std::unique_ptr<TreeNeighbor<D>>> neighbors;
  std::vector<std::vector<int>> original;
  int tags=0;
  for(int i=0;i<count;++i)tags=std::max(tags,int(input[7*i+5])+1);
  for(int k=0;k<tags;++k) {
    std::vector<int> ids;
    for(int real=1;real>=0;--real)for(int i=0;i<count;++i)
      if(int(input[7*i+5])==k and int(input[7*i+6])==real)ids.push_back(i);
    int nreal=0;for(int i:ids)nreal+=int(input[7*i+6]);
    nodes.emplace_back(new FluidNodeList<D>("support audit "+std::to_string(k),eos,nreal,ids.size()-nreal,
        hmin,hmax,.1,nperh,10000,1.e-100,1.e100));
    auto& n=*nodes.back();
    for(size_t j=0;j<ids.size();++j) {
      const auto* p=input+7*ids[j];n.positions()[j]=D::Vector(p[0],p[1]);
      n.mass()[j]=p[3];n.massDensity()[j]=p[4];
      n.Hfield()[j]=4./p[2]*D::SymTensor::one();
    }
    neighbors.emplace_back(new TreeNeighbor<D>(n,NeighborSearchType::GatherScatter,4.,D::Vector(-1000.,-1000.),D::Vector(1000.,1000.)));
    n.registerNeighbor(*neighbors.back());neighbors.back()->updateNodes();db.appendNodeList(n);original.push_back(ids);
  }
  db.updateConnectivityMap(false,false,false);
  State<D> state;StateDerivatives<D> derivs;
  auto x=db.fluidPosition();auto m=db.fluidMass();auto rho=db.fluidMassDensity();auto H=db.fluidHfield();
  state.enroll(x);state.enroll(m);state.enroll(rho);state.enroll(H);
  auto g=db.newFluidFieldList(D::Tensor::zero(),HydroFieldNames::velocityGradient);derivs.enroll(g);
  SPHSmoothingScale<D> smooth(HEvolutionType::IdealH,table());smooth.damping(damping);
  smooth.initializeProblemStartup(db);smooth.registerDerivatives(db,derivs);derivs.Zero();
  smooth.evaluateDerivatives(0.,1.,db,state,derivs);
  const auto ideal=derivs.fields(ReplaceBoundedState<D,D::SymTensor>::prefix()+HydroFieldNames::H,D::SymTensor::zero());
  const auto moment=smooth.zerothMoment();
  for(int k=0;k<tags;++k)for(size_t j=0;j<nodes[k]->numInternalNodes();++j) {
    const int i=original[k][j];auto* out=output+3*i;
    out[0]=4./ideal(k,j).xx();out[1]=moment(k,j);out[2]=table().equivalentNodesPerSmoothingScale(moment(k,j));
  }
  for(auto& n:nodes)n->unregisterNeighbor();
  return 0;
}
