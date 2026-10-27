// Isolated single-material 1D control: Eqs.63,75,76,78 and fresh CRK Q gradient.
// Native force, raw heating, RK2 and support packages are not replaced.
#include "Geometry/Dimension.hh"
#include "CRKSPH/CRKSPHBase.hh"
#include "Hydro/SpecificThermalEnergyPolicy.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Hydro/entropyWeightingFunction.hh"
#include "RK/RKCorrections.hh"
#include "RK/ReproducingKernel.hh"
#include "RK/RKFieldNames.hh"
#include "RK/computeRKSumVolume.hh"
#include "DataBase/DataBase.hh"
#include "DataBase/State.hh"
#include "DataBase/StateDerivatives.hh"
#include "DataBase/IncrementState.hh"
#include "Neighbor/PairwiseField.hh"
#include "NodeList/FluidNodeList.hh"
#include "Material/EquationOfState.hh"
#include "Boundary/Boundary.hh"
#include <vector>

namespace {
unsigned steps=0,rhsCalls=0,energyCalls=0;
std::vector<double> entropy,firstRHS;
}
extern "C" int warpsph_sod_version(){return 1;}
extern "C" unsigned warpsph_sod_steps(){return steps;}
extern "C" unsigned long long warpsph_sod_rhs_rows(){return firstRHS.size()/19;}
extern "C" const double* warpsph_sod_rhs_data(){return firstRHS.data();}

namespace Spheral {
using D=Dim<1>;using Vec=D::Vector;using Sym=D::SymTensor;using Tensor=D::Tensor;
template<> void SpecificThermalEnergyPolicy<D>::updateAsIncrement(const std::string&,State<D>&,StateDerivatives<D>&,double,double,double);

template<> void computeRKSumVolume<D>(const ConnectivityMap<D>& cm,const TableKernel<D>& W,
    const FieldList<D,Vec>& x,const FieldList<D,double>&,const FieldList<D,Sym>& H,FieldList<D,double>& V){
  VERIFY2(V.size()==1,"Single-material control requires one NodeList");V=0.;
  for(const auto& p:cm.nodePairList()){
    const auto i=p.i_node,j=p.j_node;const auto r=x(0,i)-x(0,j);
    V(0,i)+=W.kernelValue((H(0,i)*r).magnitude(),H(0,i).Determinant());
    V(0,j)+=W.kernelValue((H(0,j)*r).magnitude(),H(0,j).Determinant());
  }
  for(auto i=0u;i<V[0]->numInternalElements();++i)V(0,i)=1./(V(0,i)+W.kernelValue(0.,H(0,i).Determinant()));
}

template<> void CRKSPHBase<D>::preStepInitialize(const DataBase<D>& db,State<D>& state,StateDerivatives<D>&){
  ++steps;rhsCalls=energyCalls=0;
  VERIFY2(db.numFluidNodeLists()==1,"Exactly one physical material");
  const auto& W=state.get<ReproducingKernel<D>>(RKFieldNames::reproducingKernel(correctionOrder()));
  const auto& cm=db.connectivityMap();
  auto rho=state.fields(HydroFieldNames::massDensity,0.);
  const auto m=state.fields(HydroFieldNames::mass,0.);const auto V=state.fields(HydroFieldNames::volume,0.);
  const auto x=state.fields(HydroFieldNames::position,Vec::zero());const auto H=state.fields(HydroFieldNames::H,Sym::zero());
  auto c0=state.fields(RKFieldNames::rkCorrections(RKOrder::ZerothOrder),RKCoefficients<D>());
  auto c=state.fields(RKFieldNames::rkCorrections(correctionOrder()),RKCoefficients<D>());
  W.computeCorrections(cm,V,x,H,false,c0,c);
  for(auto* bc:this->boundaryConditions()){bc->applyFieldListGhostBoundary(c0);bc->applyFieldListGhostBoundary(c);}
  for(auto* bc:this->boundaryConditions())bc->finalizeGhostBoundary();
  auto num=db.newFluidFieldList(0.,"sod rho numerator"),den=db.newFluidFieldList(0.,"sod rho denominator");
  for(const auto& p:cm.nodePairList()){
    const auto i=p.i_node,j=p.j_node;const auto r=x(0,i)-x(0,j);
    const double wi=W.evaluateKernel(r,H(0,j),c(0,i)),wj=W.evaluateKernel(-r,H(0,i),c(0,j));
    num(0,i)+=m(0,j)*V(0,j)*wi;den(0,i)+=V(0,j)*V(0,j)*wi;
    num(0,j)+=m(0,i)*V(0,i)*wj;den(0,j)+=V(0,i)*V(0,i)*wj;
  }
  for(auto i=0u;i<rho[0]->numInternalElements();++i){
    const double w=W.evaluateKernel(Vec::zero(),H(0,i),c(0,i));
    rho(0,i)=(num(0,i)+m(0,i)*V(0,i)*w)/(den(0,i)+V(0,i)*V(0,i)*w);
  }
  for(auto* bc:this->boundaryConditions())bc->applyFieldListGhostBoundary(rho);
  for(auto* bc:this->boundaryConditions())bc->finalizeGhostBoundary();
  auto P=state.fields(HydroFieldNames::pressure,0.),cs=state.fields(HydroFieldNames::soundSpeed,0.),s=state.fields(HydroFieldNames::entropy,0.);
  const auto u=state.fields(HydroFieldNames::specificThermalEnergy,0.);
  const auto& n=dynamic_cast<const FluidNodeList<D>&>(rho[0]->nodeList());
  n.pressure(*P[0]);n.soundSpeed(*cs[0]);n.equationOfState().setEntropy(*s[0],*rho[0],*u[0]);
}

template<> bool RKCorrections<D>::initialize(const double,const double,const DataBase<D>& db,State<D>& state,StateDerivatives<D>&){
  auto V=state.fields(HydroFieldNames::volume,0.);const auto m=state.fields(HydroFieldNames::mass,0.);
  const auto x=state.fields(HydroFieldNames::position,Vec::zero());const auto H=state.fields(HydroFieldNames::H,Sym::zero());
  computeRKSumVolume<D>(db.connectivityMap(),mWR.at(RKOrder::LinearOrder).kernel(),x,m,H,V);
  for(auto* bc:this->boundaryConditions())bc->applyFieldListGhostBoundary(V);
  for(auto* bc:this->boundaryConditions())bc->finalizeGhostBoundary();
  if(mUpdateInStep)updateCorrections(db,state);
  const auto v=state.fields(HydroFieldNames::velocity,Vec::zero());
  auto g=state.fields(HydroFieldNames::ArtificialViscosityVelocityGradient,Tensor::zero());g.Zero();
  const auto c=state.fields(RKFieldNames::rkCorrections(RKOrder::LinearOrder),RKCoefficients<D>());
  const auto& W=state.get<ReproducingKernel<D>>(RKFieldNames::reproducingKernel(RKOrder::LinearOrder));
  for(const auto& p:db.connectivityMap().nodePairList()){
    const auto i=p.i_node,j=p.j_node;const auto r=x(0,i)-x(0,j),dv=v(0,i)-v(0,j);
    g(0,i)-=V(0,j)*dv.dyad(W.evaluateGradient(r,H(0,j),c(0,i)));
    g(0,j)+=V(0,i)*dv.dyad(W.evaluateGradient(-r,H(0,i),c(0,j)));
  }
  for(auto* bc:this->boundaryConditions())bc->applyFieldListGhostBoundary(g);
  for(auto* bc:this->boundaryConditions())bc->finalizeGhostBoundary();
  const auto s=state.fields(HydroFieldNames::entropy,0.);entropy.resize(s[0]->numElements());
  for(auto i=0u;i<entropy.size();++i)entropy[i]=s(0,i);
  ++rhsCalls;return true;
}

template<> void CRKSPHBase<D>::finalizeDerivatives(const double,const double,const DataBase<D>&,
    const State<D>& state,StateDerivatives<D>& derivs)const{
  auto a=derivs.fields(HydroFieldNames::hydroAcceleration,Vec::zero());
  auto du=derivs.fields(IncrementState<D,double>::prefix()+HydroFieldNames::specificThermalEnergy,0.);
  if(compatibleEnergyEvolution()){
    for(auto* bc:this->boundaryConditions()){bc->applyFieldListGhostBoundary(a);bc->applyFieldListGhostBoundary(du);}
    for(auto* bc:this->boundaryConditions())bc->finalizeGhostBoundary();
  }
  if(steps==1){
    const auto x=state.fields(HydroFieldNames::position,Vec::zero()),v=state.fields(HydroFieldNames::velocity,Vec::zero());
    const auto H=state.fields(HydroFieldNames::H,Sym::zero());const auto g=state.fields(HydroFieldNames::ArtificialViscosityVelocityGradient,Tensor::zero());
    const auto m=state.fields(HydroFieldNames::mass,0.),rho=state.fields(HydroFieldNames::massDensity,0.),u=state.fields(HydroFieldNames::specificThermalEnergy,0.);
    const auto P=state.fields(HydroFieldNames::pressure,0.),V=state.fields(HydroFieldNames::volume,0.),cs=state.fields(HydroFieldNames::soundSpeed,0.);
    const auto c=state.fields(RKFieldNames::rkCorrections(RKOrder::LinearOrder),RKCoefficients<D>());
    for(auto i=0u;i<x[0]->numElements();++i){
      const double row[]={double(rhsCalls),double(i),x(0,i).x(),4./H(0,i).xx(),m(0,i),rho(0,i),u(0,i),P(0,i),V(0,i),
          v(0,i).x(),cs(0,i),g(0,i).xx(),du(0,i),a(0,i).x(),c(0,i)[0],c(0,i)[1],c(0,i)[2],c(0,i)[3],double(i<x[0]->numInternalElements())};
      firstRHS.insert(firstRHS.end(),std::begin(row),std::end(row));
    }
  }
}

template<> void SpecificThermalEnergyPolicy<D>::update(const std::string& key,State<D>& state,
    StateDerivatives<D>& derivs,const double multiplier,const double t,const double dt){
  if(++energyCalls==1){this->updateAsIncrement(key,state,derivs,multiplier,t,dt);return;}
  auto u=state.fields(HydroFieldNames::specificThermalEnergy,0.);const auto m=state.fields(HydroFieldNames::mass,0.);
  const auto v=state.fields(HydroFieldNames::velocity,Vec::zero()),a=derivs.fields(HydroFieldNames::hydroAcceleration,Vec::zero());
  const auto& pa=derivs.get<PairwiseField<D,Vec>>(HydroFieldNames::pairAccelerations);
  const auto& pairs=mDataBasePtr->connectivityMap().nodePairList();auto rate=mDataBasePtr->newFluidFieldList(0.,"sod compatible work");
  for(auto p=0u;p<pairs.size();++p){
    const auto i=pairs[p].i_node,j=pairs[p].j_node;
    const double w=(v(0,j)-v(0,i)+.5*multiplier*(a(0,j)-a(0,i))).dot(pa[p]);
    const double f=entropyWeighting(entropy.at(i),entropy.at(j),w);
    rate(0,i)+=f*w;rate(0,j)+=(1-f)*w*m(0,i)/m(0,j);
  }
  for(auto i=0u;i<u[0]->numInternalElements();++i)u(0,i)+=multiplier*rate(0,i);
}
template<> void SpecificThermalEnergyPolicy<D>::updateAsIncrement(const std::string& key,State<D>& state,
    StateDerivatives<D>& derivs,const double multiplier,const double t,const double dt){
  std::string field,node;StateBase<D>::splitFieldKey(key,field,node);auto u=state.fields(field,0.);IncrementState<D,double> policy;
  for(auto f:u)policy.updateAsIncrement(State<D>::key(*f),state,derivs,multiplier,t,dt);
}
}
