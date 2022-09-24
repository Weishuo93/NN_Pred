c  ---------------------------------------------------------------------------
c  CFL3D is a structured-grid, cell-centered, upwind-biased, Reynolds-averaged
c  Navier-Stokes (RANS) code. It can be run in parallel on multiple grid zones
c  with point-matched, patched, overset, or embedded connectivities. Both
c  multigrid and mesh sequencing are available in time-accurate or
c  steady-state modes.
c
c  Copyright 2001 United States Government as represented by the Administrator
c  of the National Aeronautics and Space Administration. All Rights Reserved.
c 
c  The CFL3D platform is licensed under the Apache License, Version 2.0 
c  (the "License"); you may not use this file except in compliance with the 
c  License. You may obtain a copy of the License at 
c  http://www.apache.org/licenses/LICENSE-2.0. 
c 
c  Unless required by applicable law or agreed to in writing, software 
c  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
c  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
c  License for the specific language governing permissions and limitations 
c  under the License.
c  ---------------------------------------------------------------------------
c
      subroutine twoeqnvistbc(jdim,kdim,idim,q,sj,sk,si,vol,dtj,
     + x,y,z,vist3d,vor,smin,zksav,turre,damp1,blend,timestp,fnu,bx,bx2,
     + cx,cx2,dx,dx2,fx,fx2,workx,by,by2,cy,cy2,dy,dy2,fy,fy2,worky,
     + bz,bz2,cz,cz2,dz,dz2,fz,fz2,workz,ntime,tj0,tk0,ti0,nbl,
     + qj0,qk0,qi0,vj0,vk0,vi0,blank,iover,sumn1,sumn2,negn1,negn2,
     + ux,rhside,zksav2,v3dtmp,cmuv,bcj,bck,bci,nbci0,nbcidim,
     + nbcj0,nbcjdim,nbck0,nbckdim,ibcinfo,jbcinfo,kbcinfo,
     + maxbl,maxseg,volj0,volk0,voli0,nou,bou,nbuf,ibufdim,iex,iex2,
     + iex3,
     + dkdx,dkdy,dkdz,dzdx,dzdy,dzdz,dw1dx,dw1dy,dw1dz,
     + dw2dx,dw2dy,dw2dz,dw3dx,dw3dy,dw3dz,drdx,drdy,drdz,
     + dpdx,dpdy,dpdz,
     + dkdj,dkdk,dkdi,dzdj,dzdk,dzdi,dw1dj,dw1dk,dw1di,
     + dw2dj,dw2dk,dw2di,dw3dj,dw3dk,dw3di,drdj,drdk,drdi,
     + dpdj,dpdk,dpdi,
     + w1,w2,w3,srce,vx,xlscale,fdsav,nummem,iccnum)
c
c     $Id$
c
c***********************************************************************
c     Purpose:  Compute turbulent viscosity distributions using
c     2-equation turbulence models
c     (reprogrammed by Rumsey for CFL3D Version 5 - 5/96)
c     Alterations to EASM #8,9,13,14 by Rumsey - 5/00)
c***********************************************************************
c
#   ifdef CMPLX
      implicit complex(a-h,o-z)
#   endif
c
      character*120 bou(ibufdim,nbuf)
c
      dimension nou(nbuf)
      common /des/ cdes,ides,cddes
      common /info/ title(20),rkap(3),xmach,alpha,beta,dt,fmax,nit,ntt,
     .        idiag(3),nitfo,iflagts,iflim(3),nres,levelb(5),mgflag,
     .        iconsf,mseq,ncyc1(5),levelt(5),nitfo1(5),ngam,nsm(5),iipv
      common /ivals/ p0,rho0,c0,u0,v0,w0,et0,h0,pt0,rhot0,qiv(5),
     .        tur10(7)
      common /easmv/ c10,c11,c2,c3,c4,c5,sigk1,cmuc1,ieasm_type
      common /fluid/ gamma,gm1,gp1,gm1g,gp1g,ggm1
      common /fluid2/ pr,prt,cbar
      common /lam/ ilamlo,ilamhi,jlamlo,jlamhi,klamlo,klamhi,
     .        i_lam_forcezero
      common /maxiv/ ivmx
      common /mgrd/ levt,kode,mode,ncyc,mtt,icyc,level,lglobal
      common /reyue/ reue,tinf,ivisc(3)
      common /sklton/ isklton
      common /twod/ i2d
      common /zero/ iexp
      common /wallfun/ iwf(3)
      common /turbconv/ cflturb(7),edvislim,iturbprod,nsubturb,nfreeze,
     .                  iwarneddy,itime2read,itaturb,tur1cut,tur2cut,
     .                  iturbord,tur1cutlev,tur2cutlev
      common /unst/ time,cfltau,ntstep,ita,iunst,cfltau0,cfltauMax
      common /konew/ ikoprod,isstdenom,pklimterm,ibeta8kzeta,i_bsl,
     .        keepambient,re_thetat0,i_wilcox06,i_wilcox06_chiw,
     .        i_turbprod_kterm,i_catris_kw,prod2d3dtrace,
     .        i_compress_correct,isstsf,i_wilcox98,i_wilcox98_chiw,
     .        isst2003
      common /curvat/ isarc2d,sarccr3,ieasmcc2d,isstrc,sstrc_crc,
     .        isar,crot,isarc3d
      common /easmlim/ cmulim
      common /axisym/ iaxi2plane,iaxi2planeturb,istrongturbdis,iforcev0
c
      dimension q(jdim,kdim,idim,5),sj(jdim,kdim,idim-1,5),
     + sk(jdim,kdim,idim-1,5),si(jdim,kdim,idim,5),vol(jdim,kdim,idim-1)
     +,dtj(jdim,kdim,idim-1),x(jdim,kdim,idim),y(jdim,kdim,idim),
     + z(jdim,kdim,idim),vist3d(jdim,kdim,idim),
     + vor(jdim-1,kdim-1,idim-1),smin(jdim-1,kdim-1,idim-1)
      dimension damp1(jdim-1,kdim-1,idim-1),
     + fnu(0:jdim,0:kdim,0-iex3:idim+iex3)
     + , blend(jdim-1,kdim-1,idim-1),timestp(jdim-1,kdim-1,idim-1)
      dimension bx(kdim-1,jdim-1),bx2(kdim-1,jdim-1),
     + cx(kdim-1,jdim-1),cx2(kdim-1,jdim-1),dx(kdim-1,jdim-1),
     + dx2(kdim-1,jdim-1),fx(kdim-1,jdim-1),fx2(kdim-1,jdim-1),
     + workx(kdim-1,jdim-1),
     +          by(jdim-1,kdim-1),by2(jdim-1,kdim-1),
     + cy(jdim-1,kdim-1),cy2(jdim-1,kdim-1),dy(jdim-1,kdim-1),
     + dy2(jdim-1,kdim-1),fy(jdim-1,kdim-1),fy2(jdim-1,kdim-1),
     + worky(jdim-1,kdim-1),
     +          bz(kdim-1,idim-1),bz2(kdim-1,idim-1),
     + cz(kdim-1,idim-1),cz2(kdim-1,idim-1),dz(kdim-1,idim-1),
     + dz2(kdim-1,idim-1),fz(kdim-1,idim-1),fz2(kdim-1,idim-1),
     + workz(kdim-1,idim-1)
      dimension turre(0-iex:jdim+iex,0-iex:kdim+iex,0-iex2:idim+iex2,2),
     + blank(jdim,kdim,idim),
     + zksav(jdim,kdim,idim,nummem),rhside(jdim-1,kdim-1,idim-1,2)
     + ,tj0(kdim,idim-1,nummem,4),tk0(jdim,idim-1,nummem,4),
     +  ti0(jdim,kdim,nummem,4),
     + qj0(kdim,idim-1,5,4),qk0(jdim,idim-1,5,4),qi0(jdim,kdim,5,4)
     +,vj0(kdim,idim-1,1,4),vk0(jdim,idim-1,1,4),vi0(jdim,kdim,1,4)
      dimension ux(jdim-1,kdim-1,idim-1,9),
     + zksav2(jdim,kdim,idim,2*nummem),
     + v3dtmp(0:jdim,0:kdim,0-iex3:idim+iex3)
      dimension cmuv(jdim-1,kdim-1,idim-1)
      dimension bcj(kdim,idim-1,2),bck(jdim,idim-1,2),bci(jdim,kdim,2)
      dimension nbci0(maxbl),nbcidim(maxbl),nbcj0(maxbl),nbcjdim(maxbl),
     +          nbck0(maxbl),nbckdim(maxbl),ibcinfo(maxbl,maxseg,7,2),
     +          jbcinfo(maxbl,maxseg,7,2),kbcinfo(maxbl,maxseg,7,2)
      dimension volj0(kdim,idim-1,4),
     +          volk0(jdim,idim-1,4),voli0(jdim,kdim,4)
      dimension  dkdx(jdim),dkdy(jdim),dkdz(jdim),
     +           dzdx(jdim),dzdy(jdim),dzdz(jdim),
     +           dw1dx(jdim),dw1dy(jdim),dw1dz(jdim),
     +           dw2dx(jdim),dw2dy(jdim),dw2dz(jdim),
     +           dw3dx(jdim),dw3dy(jdim),dw3dz(jdim),
     +           drdx(jdim),drdy(jdim),drdz(jdim),
     +           dpdx(jdim),dpdy(jdim),dpdz(jdim)
      dimension dkdj(jdim),dkdk(jdim),dkdi(jdim),
     +          dzdj(jdim),dzdk(jdim),dzdi(jdim),
     +          dw1dj(jdim),dw1dk(jdim),dw1di(jdim),
     +          dw2dj(jdim),dw2dk(jdim),dw2di(jdim),
     +          dw3dj(jdim),dw3dk(jdim),dw3di(jdim),
     +          drdj(jdim),drdk(jdim),drdi(jdim),
     +          dpdj(jdim),dpdk(jdim),dpdi(jdim)
      dimension w1(0:jdim,0:kdim,0:idim),
     +          w2(0:jdim,0:kdim,0:idim),
     +          w3(0:jdim,0:kdim,0:idim)
      dimension srce(0:jdim,0:kdim,0:idim,2)
      dimension vx(0:jdim,0:kdim,idim-1,iccnum)
      dimension xlscale(jdim-1,kdim-1,idim-1),
     +          fdsav(jdim-1,kdim-1,idim-1)
c
c
c   Variables:
c
c      jdim,kdim,idim - dimensions of this block
c      q - primitive variables (rho, u, v, w, p)
c      sj,sk,si - metric terms (defined on cell FACES)
c      vol - cell volume
c      volj0,volk0,voli0 - ghost-cell volumes
c      dtj - vol/dt
c      x,y,z - grid vertex locations
c      vist3d - turbulent eddy viscosity (nondimensionalized by mulamref)
c      vor - vorticity
c      smin - minimum distance to any wall - set negative if laminar region!
c      zksav - saved values of the 2 turbulent quantities (1=omega or epsilon,
c              2=k)
c                omega is nondimensionalized by rhoref*aref**2/mulamref
c                epsilon          "             rhoref*aref**4/mulamref
c                k                "             aref**2
c      zksav2 - auxiliary k and omega (or epsilon) for saving quantities
c               from last time step during time-accurate subiterations
c               (k,j,i,1) and (j,k,i,2) = k and omega (or epsilon)
c               (k,j,i,3) and (j,k,i,4) = delta k and delta omega (or epsilon)
c      turre - working values of 2 turb quantities in this subroutine
c      damp1 - for SST=CD (cross-derivative) term; for others, sometimes used
c              as storage for linearizations of production terms for addition
c              to LHS diagonal elements
c      blend - blending term for 2-layer models (just SST right now); 
c              =0 Wilcox06
c              =1 otherwise
c      timestp - time step array delta t
c      fnu - laminar viscosity (from Sutherland's law), nondim by mulamref
c      bx,bx2,cx,cx2,dx,dx2,fx,fx2 - sub, diag, superdiag, & RHS in 
c                                    eta direction
c      by,by2,cy,cy2,dy,dy2,fy,fy2 - sub, diag, superdiag, & RHS in 
c                                    xi direction
c      bz,bz2,cz,cz2,dz,dz2,fz,fz2 - sub, diag, superdiag, & RHS in 
c                                    zeta direction
c      workx,worky,workz - work arrays for tridiagonal solvers
c      ntime - time counter
c      tj0,tk0,ti0 - BCs for turbulent quantities (1=omega or epsilon, 2=k)
c      nbl = block number currently working on
c      qj0,qk0,qi0 - BCs for q's
c      vj0,vk0,vi0 - BCs for vist3d
c      blank - iblanking array for overset
c      iover - overset gridding parameter
c      sumn1,sumn2 - residual for 2 turbulence equations
c      negn1,negn2 - number of locations where the solution yields a "negative"
c                    turbulence quantity.  This SHOULD be zero, but it seems
c                    to be OK if there are only a few of these.  When they
c                    go negative, the values are artificially limited to be
c                    > 0.  When these numbers are large, it indicates that
c                    the solution is probably going to blow up.  Check your
c                    grid for excessive grid stretching, or try lowering CFL.
c      ux - 9 components of velocity derivative: ux,uy,uz,vx,vy,vz,wx,wy,wz 
c           at cell centers.  Used for nonlinear models only.
c      rhside - right-hand-side terms for 2 eqns
c      v3dtmp - temporary storage for vist3d (needed for nonlinear models,
c               which require turb viscosity in diffusion term WITHOUT
c               variable cmu)
c      cmuv   - storage for variable cmu in model #13
c      bcj,bck,bci - =0 for cell-center BC, =1 for at-face-center BC
c      nbci0,nbcidim,nbcj0,nbcjdim,nbck0,nbckdim - no. of BC segments each face
c      ibcinfo,jbcinfo,kbcinfo - gives i,j,k start and end indices for each
c               BC segment, among other things
c      maxbl,maxseg - dimensions used in nbci0,etc and ibcinfo, etc arrays
c      xlscale - length scale for use with DES 2-eqn model
c      fdsav - f_d parameter for use with DDES
c
      if(isklton .gt. 0) then
         nou(1) = min(nou(1)+1,ibufdim)
         write(bou(nou(1),1),'(''     Computing turbulent'',
     +'' viscosity using 2-eqns, block='',i5)') nbl
         nou(1) = min(nou(1)+1,ibufdim)
         write(bou(nou(1),1),'(''     Freestream tur10,tur20 = '',
     +     2e19.8)') real(tur10(1)),real(tur10(2))
        if(iturbord .eq. 1) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     1st order advection on RHS'')')
        else
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     2nd order advection on RHS'')')
        end if
      end if
      if(isklton .gt. 0) then
        if(ivmx .eq. 6) then
           if(i_wilcox06 .eq. 1) then
             if(i_wilcox06_chiw .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     k-omega (Wilcox 06),'',
     +       '' with vortex stretching term'')')
             else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     k-omega (Wilcox 06),'',
     +       '' without vortex stretching term'')')
             end if
           elseif(i_wilcox98 .eq. 1) then
             if(i_wilcox98_chiw .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     k-omega (Wilcox 98),'',
     +       '' with vortex stretching term'')')
             else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     k-omega (Wilcox 98),'',
     +       '' without vortex stretching term'')')
             end if
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     k-omega (Wilcox 88)'')')
           end if
           if(ikoprod .eq. 1) then
             if(i_turbprod_kterm .ne. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     +       '' term'')')
             else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     +       '' term with 2/3rho*k subtracted'')')
             end if
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     approx (vort) production'',
     .         '' term'')')
           end if
        end if
        if(ivmx .eq. 7 .and. isst2003 .eq. 1) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-omega SST-2003 (Menter)'')')
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''       ignores ikoprod,isstdenom'',
     .       '' (these will ALWAYS be on)'')')
        end if
        if(isst2003 .ne. 1) then
        if(ivmx .eq. 7 .and. i_bsl .ne. 1) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-omega SST (Menter)'')')
           if(ikoprod .eq. 1) then
             if(i_turbprod_kterm .ne. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     +       '' term'')')
             else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     +       '' term with 2/3rho*k subtracted'')')
             end if
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     approx (vort) production'',
     .         '' term'')')
           end if
           if(isstdenom .eq. 0) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     vort in denom of mut term'')')
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     S in denom of mut term'')')
           end if
        else if(ivmx .eq. 7 .and. i_bsl .eq. 1) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-omega BSL (Menter)'')')
           if(ikoprod .eq. 1) then
             if(i_turbprod_kterm .ne. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     +       '' term'')')
             else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     +       '' term with 2/3rho*k subtracted'')')
             end if
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     approx (vort) production'',
     .         '' term'')')
           end if
        end if
        end if
        if(ivmx .eq. 8) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-omega (EASM, '',
     .   ''linear)'')')
           if(iturbprod .eq. 0) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     .         '' term'')')
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     exact production term'')')
           end if
           if (ieasmcc2d .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     curvature correction ON'')')
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''       WARNING: curv terms are'',
     +      '' active in 2-D sense only!!!'')')
           end if
        end if
        if(ivmx .eq. 9) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-epsilon (EASM,'',
     .   ''linear)'')')
           if(iturbprod .eq. 0) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     .         '' term'')')
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     exact production term'')')
           end if
           if (ieasmcc2d .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     curvature correction ON'')')
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''       WARNING: curv terms are'',
     +      '' active in 2-D sense only!!!'')')
           end if
        end if
        if(ivmx .eq.10) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-epsilon (Abid)'')')
           if(ikoprod .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     +       '' term'')')
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     approx (vort) production'',
     .         '' term'')')
           end if
        end if
        if(ivmx .eq.11) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-epsilon (G-S EASM, '',
     .   ''nonlinear)'')')
           if (ieasmcc2d .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     curvature correction ON'')')
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''       WARNING: curv terms are'',
     +      '' active in 2-D sense only!!!'')')
           end if
        end if
        if(ivmx .eq.12) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-omega (G-S EASM, '',
     .   ''nonlinear)'')')
           if (ieasmcc2d .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     curvature correction ON'')')
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''       WARNING: curv terms are'',
     +      '' active in 2-D sense only!!!'')')
           end if
        end if
        if(ivmx .eq.13) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-epsilon (EASM,'',
     .   ''nonlinear)'')')
           if(iturbprod .eq. 0) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     .         '' term'')')
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     exact production term'')')
           end if
           if (ieasmcc2d .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     curvature correction ON'')')
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''       WARNING: curv terms are'',
     +      '' active in 2-D sense only!!!'')')
           end if
        end if
        if(ivmx .eq.14) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-omega (EASM,'',
     .   ''nonlinear)'')')
           if(iturbprod .eq. 0) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     .         '' term'')')
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     exact production term'')')
           end if
           if (ieasmcc2d .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     curvature correction ON'')')
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''       WARNING: curv terms are'',
     +      '' active in 2-D sense only!!!'')')
           end if
        end if
        if(ivmx .eq.15) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-enstrophy (Note: default '',
     .   ''tur1cut=-1 if no keyword)'')')
           if(ikoprod .eq. 2) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     exact production term'')')
           else if (ikoprod .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     .         '' term'')')
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     approx (vort) production'',
     .         '' term'')')
           end if
        end if
        if(ivmx .eq.16) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     k-kL-MEAH2015'')')
           if(ikoprod .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     strain-based production'',
     .         '' term'')')
           else
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     approx (vort) production'',
     .         '' term'')')
           end if
        end if
c
        if (ides .eq. 1) then
          nou(1) = min(nou(1)+1,ibufdim)
          write(bou(nou(1),1),'(''   using model in conjunction'',
     +      '' with DES, cdes='',f7.3)') cdes
        else if (ides .eq. 2) then
          nou(1) = min(nou(1)+1,ibufdim)
          write(bou(nou(1),1),'(''   using model in conjunction'',
     +      '' with DDES, cdes='',f7.3)') cdes
        else if (ides .eq. 3) then
          nou(1) = min(nou(1)+1,ibufdim)
          write(bou(nou(1),1),'(''   using model in conjunction'',
     +      '' with MDDES, cdes='',f7.3,'', cddes='',f7.3)') cdes,cddes
        end if
c
        if(ikoprod .eq. 1 .and. (ivmx.eq.6.or.ivmx.eq.7.or.
     .    ivmx.eq.10)) then
        if(abs(real(prod2d3dtrace)-0.5) .lt. 0.001) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     Sij used in 2SijSij prod'',
     .         '' term forced to be traceless in 2-D sense'')')
        else if(abs(real(prod2d3dtrace)-0.33333333) .lt. 0.001) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     Sij used in 2SijSij prod'',
     .         '' term forced to be traceless in 3-D sense'')')
        end if
        end if
        if(ivmx.eq.6.and.i_wilcox06.eq.1) then
        if(abs(real(prod2d3dtrace)-0.5) .lt. 0.001) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     Sij used in 2SijSij in W06'',
     .        '' stress-limiter forced to be traceless in 2-D sense'')')
        else if(abs(real(prod2d3dtrace)-0.33333333) .lt. 0.001) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     Sij used in 2SijSij in W06'',
     .        '' stress-limiter forced to be traceless in 3-D sense'')')
        end if
        end if
c
        if(ivmx .eq. 9 .or. ivmx .eq.10 .or. ivmx .eq.11 .or.
     .     ivmx .eq. 13.or. ivmx .eq .15) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     WARNING: k-epsilon and '',
     .     ''k-enstrophy models sometimes fail to go turbulent!'')')
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''         We recommend either'',
     .     '' restarting from a different converged model,'')')
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''         or, if starting from'',
     .     '' scratch, freezing the model to its initialzed'')')
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''         levels (using keyword'',
     .     '' NFREEZE) until the flowfield is converged enough'')')
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''         for turbulence to'',
     .     '' sustain itself'')')
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     ALWAYS check vist3d levels to'',
     .     '' insure turbulence has tripped!'')')
           if (ikoprod .ne. 0) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     Also check that there is'',
     .       '' not excessive turbulence at stagnation regions.'')')
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''         Try ikoprod=0 if'',
     .       '' there is.'')')
           end if
        end if
        if(ivmx .eq. 9 .or. ivmx .eq.10 .or. ivmx .eq.11 .or.
     .     ivmx .eq. 13) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     Also, k-e models are NOT'',
     .     '' RECOMMENDED for'')')
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''         adverse-pressure-gradient'',
     .     '' wall-bounded flows.'')')
           if (iturbord .eq. 2) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     WARNING: 2nd order'',
     .       '' advection (iturbord=2)'')')
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''          can converge poorly'',
     .       '' for k-epsilon models'')')
           end if
        end if
        if(ivmx .eq. 11 .or. ivmx .eq. 12 .or. ivmx .eq. 13 .or.
     .     ivmx .eq. 14) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     WARNING: when using nonlinear'',
     .     '' models, the grid'')')
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''         needs to have min y+ of'',
     .     '' order 1.  If it is much larger,'')')
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''         you may need'',
     .     '' WallFunctions to keep from blowing up!'')')
        end if
        if(ivmx .eq. 6 .or. ivmx .eq. 7 .or. ivmx .eq. 8 .or.
     .     ivmx .eq. 12.or. ivmx .eq.14) then
           if (i_catris_kw .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     Catris compressible '',
     .       ''correction for turb diffusion terms being used'')')
           end if
        end if
        if(ivmx .eq. 6 .or. ivmx .eq. 7) then
           if (i_compress_correct .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     Wilcox-type dilatation-'',
     .       ''dissipation compressibility correction employed'')')
           else if (i_compress_correct .eq. 2) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     Zeman-BL-type dilatation-'',
     .       ''dissipation compressibility correction employed'')')
           end if
           if (isstrc .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     SSTRC-type curvature'',
     .       '' correction employed (AIAA 98-2554), sstrc_crc='',f5.2)')
     .       sstrc_crc
           end if
           if (isstrc .eq. 2) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     SSTRC-type curvature'',
     .       '' correction employed (Smirnov & Menter)'')')
           end if
           if (isstsf .eq. 1) then
             nou(1) = min(nou(1)+1,ibufdim)
             write(bou(nou(1),1),'(''     SST-sf separation fix'',
     .       '' correction employed'')')
           end if
        end if
c
        if(keepambient .eq. 1 .and. ivmx .ne. 15) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     2-eqn ambient turbulence'',
     .     '' levels not allowed to decay'')')
        end if
c
        if(iaxi2planeturb .eq. 1) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     2-eqn model ignoring i-dir'')')
        end if
        if(istrongturbdis .eq. 1) then
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'(''     strong conserv - diss terms'')')
        end if
c
      end if
c
c   Note: (10.**(-iexp) is machine zero)
      xminn=10.**(-iexp+1)
c
      catris_kw=0.0
      if (i_catris_kw .eq. 1) catris_kw=1.0
      if (i_compress_correct .eq. 1) then
c        Wilcox-type (Turb Modeling for CFD, Ed 3, 2006, p. 258)
         turb_mach_0 = 0.25
         xsi_star = 2.0
         gam_comp = 0.0
      else if (i_compress_correct .eq. 2) then
c        Zeman B.L.-type (AIAA 93-0897)
         turb_mach_0 = 0.2
         xsi_star = 0.75
         gam_comp = 0.66
      end if
c
c Set number of subiterations to solve turbulence field eqn per iteration
c (usually, 1 is sufficient... but if residual diverges then may need more)
c
      nsubit=nsubturb
c
c Set factors that multiply the N-S CFL number, to determine CFL number
c for the turbulence model (this typically can be 2 - 10... optimum value
c is a function of the case).  If turb model seems to be having trouble
c converging, lowering these factors may be one strategy to try. NOTE: factors
c used only for steady state cases, not time accurate cases.
c They can be overridden with keyword "cflturb" if want all CFLs to be the
c same (not recommended), or by cflturb1, cflturb2, etc to set individual
c ones.
c
      if(ivmx .eq. 6) factor=2.
      if(ivmx .eq. 7) factor=10.
      if(ivmx .eq. 8 .or. ivmx .eq. 12 .or. ivmx .eq. 14) factor=5.
      if(ivmx .eq. 10 .or. ivmx .eq. 11) factor=2.
      if(ivmx .eq. 9 .or. ivmx .eq. 13) factor=5.
      if(ivmx .eq. 15) factor=10.
      if(ivmx .eq. 16) factor=10.
      factor1=factor
      factor2=factor
c
c Overwrite factors with keyword value "cflturb()" if nonzero
c
      if (real(cflturb(1)).ne.0.) then
         factor1 = cflturb(1)
      end if
      if (real(cflturb(2)).ne.0.) then
         factor2 = cflturb(2)
      end if
c factor2 is set relative to factor1
      factor2=factor2/factor1
c
c Timestep for turb model
c
      if (real(dt).lt.0) then
         do i=1,idim-1
         do k=1,kdim-1
         do j=1,jdim-1
           timestp(j,k,i)=factor1*vol(j,k,i)/dtj(j,k,i)
           timestp(j,k,i)=ccmincr(timestp(j,k,i),100.)
         enddo
         enddo
         enddo
      else
c          turbulence model advanced with physical time only
c          (pseudo-time term NOT included, even for tau-TS in mean-
c          flow equations, since multigrid is not used for turb. eq.)
         do i=1,idim-1
         do k=1,kdim-1
         do j=1,jdim-1
           timestp(j,k,i)=dt
           factor2=1.
         enddo
         enddo
         enddo
      end if
c
c Set up constants
      vk =.41
      a1 =.31
c The following will effectively override the SST denominator option:
      if (i_bsl .eq. 1) a1 = 1.e8
c Constants for Set 1:
c  cmuc1 (normally 0.09):
      cmuc1=0.09
c  constants for EASM models:
      if(ivmx.eq.8 .or. ivmx.eq.11 .or. ivmx .eq. 12 .or.
     .   ivmx.eq.9 .or. ivmx.eq.13 .or. ivmx .eq. 14) then
c     SSG (linearized... not fn of bij^2; Sij coeff simplified):
        cmuc1=0.081
        if(ivmx .eq. 9 .or. ivmx .eq. 13) cmuc1=0.0885
        if(ivmx .eq. 8 .or. ivmx .eq. 14) cmuc1=0.0895
c       Note: c1 = 2*c10; c5 and gg only used for ivmx=11,12
c       Note: the following choices (other than ieasm_type=0) are NOT
c       official models (they are experimental only), because only 
c       pressure-strain is changed, and not other constants in the 
c       k-eps or k-omega eqns
        if (ieasm_type .eq. 1) then
c         Wallin-Johansson type pressure-strain (close to LRR-QI)
          c10=3.6
          c11=0.
          c2=0.8
          constt=5./9.
          c3=6./11.*(2.+(3.*constt))
          c4=2./11.*(10.-(7.*constt))
        else if (ieasm_type .eq. 2) then
c         LRR-QI type pressure-strain
          c10=3.0
          c11=0.
          c2=0.8
          constt=0.4
          c3=6./11.*(2.+(3.*constt))
          c4=2./11.*(10.-(7.*constt))
        else if (ieasm_type .eq. 3 .or. ieasm_type .eq. 4) then
          c10=3.4
          c11=1.8
          c2=1.2
          c3=1.25
          c4=0.40
          css=0.84
        else
c         linear SSG Gatski-Rumsey type (default) pressure-strain
          c10=3.4
          c11=1.8
          c2=0.36
          c3=1.25
          c4=0.40
          c5=1.88
          css=0.0
          gg=1./(c10+c5-1.)
        end if
c       Durbin TCFD 1991 near-wall limiter (0=off)
c       (using it tends to delay separation - generally not desired!)
        idurbinlim=0
      end if
c  Note: for i_wilcox06, beta1, sigo1, sigk1, alp1 not used
c  (only beta2, sigo2, sigk2, alp2 are used)
c  beta1 (constant in omega or epsilon destruction term):
      if(ivmx.eq.6 .and. i_wilcox98.eq.1)           beta1=0.072
      if(ivmx.eq.6 .and. i_wilcox98.ne.1)           beta1=0.075
      if(ivmx.eq.7)                                 beta1=0.075
      if(ivmx.eq.12)                                beta1=0.83
      if(ivmx.eq.8 .or. ivmx.eq.14)                 beta1=0.83
      if(ivmx.eq.10 .or. ivmx.eq.11 .or.ivmx.eq.9
     .              .or. ivmx.eq.13)                beta1=1.83
c  sigo1 (constant in omega, epsilon, or zeta diffusion term):
      if(ivmx.eq.6 .or. ivmx.eq.7)                  sigo1=0.5
      if(ivmx.eq.12)                                sigo1=1./2.20
      if(ivmx.eq.10)                                sigo1=0.71429
      if(ivmx.eq.11)                                sigo1=1./1.3
      if(ivmx.eq.15)                                sigo1=1.46
c
c  sigk1 (constant in k diffusion term):
      if(ivmx.eq.6)                                 sigk1=0.5
      if(ivmx.eq.7 .and. i_bsl .ne. 1)              sigk1=0.85
      if(ivmx.eq.7 .and. i_bsl .eq. 1)              sigk1=0.5
      if(ivmx.eq.12)                                sigk1=1./1.4
      if(ivmx.eq.10 .or. ivmx.eq.11 .or.ivmx.eq.9
     .              .or. ivmx.eq.13)                sigk1=1.00
      if(ivmx.eq.8 .or. ivmx.eq.14)                 sigk1=1./1.0
      if(ivmx.eq.15)                                sigk1 = 1.8
c
c  alp1 (constant in omega or epsilon production term):
      if(ivmx.eq.6 .and. i_wilcox98.eq.1)           alp1 = 13./25.
      if(ivmx.eq.6 .and. i_wilcox98.ne.1)           alp1 =
     +    beta1/cmuc1 - sigo1*vk*vk/sqrt(cmuc1)
      if(ivmx.eq.7 .and. isst2003.ne.1)             alp1 =
     +    beta1/cmuc1 - sigo1*vk*vk/sqrt(cmuc1)
      if(ivmx.eq.7 .and. isst2003.eq.1)             alp1 = 5./9.
      if(ivmx.eq.10)                                alp1 =1.45
      if(ivmx.eq.11 .or. ivmx.eq.12)                alp1 =
     +    beta1 - vk*vk*sigo1/sqrt(cmuc1)
      if(ivmx.eq.9 .or. ivmx.eq.13)                 alp1 = 1.44
      if(ivmx.eq.8 .or. ivmx.eq.14)                 alp1 = 0.53
      if(ivmx.eq.9 .or. ivmx.eq.13)                 sigo1=
     + sqrt(cmuc1)*(beta1-alp1)/(vk*vk)
      if(ivmx.eq.8 .or. ivmx.eq.14)                 sigo1=
     + sqrt(cmuc1)*(beta1-alp1)/(vk*vk)
      if ((ivmx.eq.8 .or. ivmx.eq.14) .and. (ieasm_type.eq.3 .or.
     +    ieasm_type.eq.4)) sigo1=0.65
c
c Coefficient for fnu in diffusion term
      sigkmu = 1.0
      if(ivmx.eq.15) then
        sigkmu = 1./3.
      endif
c
c Constants for Set 2 (2-layer SST model and Wilcox06 only):
      if (ivmx.eq.6 .and. i_wilcox06.eq.1) then
        cmuc2=0.09
        beta2=0.0708
        sigo2=0.5
        sigk2=0.6
        alp2 =13./25.
        sigd0=0.125
      else
        cmuc2=0.09
        beta2=0.0828
        sigo2=0.856
        sigk2=1.00
        alp2 =beta2/cmuc2 - sigo2*vk*vk/sqrt(cmuc2)
        if (isst2003 .eq. 1) then
          alp2=0.44
        end if
      end if
c
c For k-kL-MEAH2015, set up constants here:
c   (note cmuc1 is the usual 0.09, vk is the usual 0.41)
      if (ivmx .eq. 16) then
        sigk1=1.0
        sigo1=1.0
        zeta1_kkl=1.2
        zeta2_kkl=0.97
        zeta3_kkl=0.13
        c11_kkl=10.
        c12_kkl=1.3
        cd1_kkl=4.7
      end if
c
c Set up some other needed parameters
      jd2=(jdim-1)/2
      re=reue/xmach
      c2b=cbar/tinf
      c2bp=c2b+1.0
c
      iwrite=0
c
      if (icyc .le. nfreeze) then
        if (isklton .gt. 0) then
           nss=min(ncyc,nfreeze)
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),*)
           nou(1) = min(nou(1)+1,ibufdim)
           write(bou(nou(1),1),'('' turbulence model is frozen '',
     +     ''for '',i5,'' iterations or subits'')') nss
        end if
        sumn1 = 0.
        sumn2 = 0.
        negn1 = 0
        negn2 = 0
        return
      end if
      phi=0.
      if (real(dt) .gt. 0.) then
        if (abs(ita) .eq. 2) then
          phi=0.5
        else
          phi=0.
        end if
c   revert to old way (always 1st order for turb model) if itaturb=0
        if (itaturb .eq. 0) then
          phi=0.
          if (isklton .gt. 0) then
            nou(1) = min(nou(1)+1,ibufdim)
            write(bou(nou(1),1),'(''   turb model is 1st'',
     +       '' order in time'')')
          end if
        else
          if (isklton .gt. 0) then
            nou(1) = min(nou(1)+1,ibufdim)
            write(bou(nou(1),1),'(''   turb model is same'',
     +       '' order in time as mean flow eqns'')')
          end if
        end if
      end if
c
c Get laminar viscosity at cell centers
      do i=1,idim-1
        do k=1,kdim-1
          do j=1,jdim-1
            tt=gamma*q(j,k,i,5)/q(j,k,i,1)
            fnu(j,k,i)=c2bp*tt*sqrt(tt)/(c2b+tt)
          enddo
        enddo
      enddo
      do i=1,idim-1
        do k=1,kdim-1
          tt=gamma*qj0(k,i,5,1)/qj0(k,i,1,1)
          fnu(0,k,i)=c2bp*tt*sqrt(tt)/(c2b+tt)
          tt=gamma*qj0(k,i,5,3)/qj0(k,i,1,3)
          fnu(jdim,k,i)=c2bp*tt*sqrt(tt)/(c2b+tt)
        enddo
      enddo
      do i=1,idim-1
        do j=1,jdim-1
          tt=gamma*qk0(j,i,5,1)/qk0(j,i,1,1)
          fnu(j,0,i)=c2bp*tt*sqrt(tt)/(c2b+tt)
          tt=gamma*qk0(j,i,5,3)/qk0(j,i,1,3)
          fnu(j,kdim,i)=c2bp*tt*sqrt(tt)/(c2b+tt)
        enddo
      enddo
      if (i2d .ne. 1 .and. iaxi2planeturb .ne. 1) then
      do j=1,jdim-1
        do k=1,kdim-1
          tt=gamma*qi0(j,k,5,1)/qi0(j,k,1,1)
          fnu(j,k,0)=c2bp*tt*sqrt(tt)/(c2b+tt)
          tt=gamma*qi0(j,k,5,3)/qi0(j,k,1,3)
          fnu(j,k,idim)=c2bp*tt*sqrt(tt)/(c2b+tt)
        enddo
      enddo
      end if
c
c Load appropriate turb viscosity at cell centers (NOTE:  the code still
c uses vi0,vj0,vk0 values for turb viscosity at ghost cells - this means
c that for these particular cases it is using slightly different form
c at the block edges)
      if (ivmx .eq. 12) then
      do i=1,idim-1
        do k=1,kdim-1
          do j=1,jdim-1
            v3dtmp(j,k,i)=cmuc1*q(j,k,i,1)*zksav(j,k,i,2)/
     +                    zksav(j,k,i,1)
            v3dtmp(j,k,i)=ccmin(v3dtmp(j,k,i),edvislim)
          enddo
        enddo
      enddo
      else if (ivmx .eq. 11 .or. ivmx .eq. 9 .or. ivmx .eq. 13) then
      do i=1,idim-1
        do k=1,kdim-1
          do j=1,jdim-1
            v3dtmp(j,k,i)=cmuc1*q(j,k,i,1)*zksav(j,k,i,2)**2/
     +                    zksav(j,k,i,1)
            v3dtmp(j,k,i)=ccmin(v3dtmp(j,k,i),edvislim)
          enddo
        enddo
      enddo
      else if (ivmx .eq. 6 .and. i_wilcox06 .eq. 1) then
      do i=1,idim-1
        do k=1,kdim-1
          do j=1,jdim-1
            v3dtmp(j,k,i)=q(j,k,i,1)*zksav(j,k,i,2)/
     +                    zksav(j,k,i,1)
            v3dtmp(j,k,i)=ccmin(v3dtmp(j,k,i),edvislim)
          enddo
        enddo
      enddo
      else
      do i=1,idim-1
        do k=1,kdim-1
          do j=1,jdim-1
            v3dtmp(j,k,i)=vist3d(j,k,i)
          enddo
        enddo
      enddo
      end if
c   Load appropriate vist3d value into ghost cells
      do i=1,idim-1
      do j=1,jdim-1
        v3dtmp(j,0,i)=bck(j,i,1)*(iwf(3)*v3dtmp(j,1,i) +
     +    (1-iwf(3))*2.*vk0(j,i,1,1)-v3dtmp(j,1,i))+
     +    (1.-bck(j,i,1))*vk0(j,i,1,1)
        v3dtmp(j,kdim,i)=bck(j,i,2)*(iwf(3)*v3dtmp(j,kdim-1,i) +
     +    (1-iwf(3))*2.*vk0(j,i,1,3)-v3dtmp(j,kdim-1,i))+
     +    (1.-bck(j,i,2))*vk0(j,i,1,3)
      enddo
      enddo
      do i=1,idim-1
      do k=1,kdim-1
        v3dtmp(0,k,i)=bcj(k,i,1)*(iwf(2)*v3dtmp(1,k,i) +
     +    (1-iwf(2))*2.*vj0(k,i,1,1)-v3dtmp(1,k,i))+
     +    (1.-bcj(k,i,1))*vj0(k,i,1,1)
        v3dtmp(jdim,k,i)=bcj(k,i,2)*(iwf(2)*v3dtmp(jdim-1,k,i) +
     +    (1-iwf(2))*2.*vj0(k,i,1,3)-v3dtmp(jdim-1,k,i))+
     +    (1.-bcj(k,i,2))*vj0(k,i,1,3)
      enddo
      enddo
      if (i2d .ne. 1 .and. iaxi2planeturb .ne. 1) then
      do j=1,jdim-1
      do k=1,kdim-1
        v3dtmp(j,k,0)=bci(j,k,1)*(iwf(1)*v3dtmp(j,k,1) +
     +    (1-iwf(1))*2.*vi0(j,k,1,1)-v3dtmp(j,k,1))+
     +    (1.-bci(j,k,1))*vi0(j,k,1,1)
        v3dtmp(j,k,idim)=bci(j,k,2)*(iwf(1)*v3dtmp(j,k,idim-1) +
     +    (1-iwf(1))*2.*vi0(j,k,1,3)-v3dtmp(j,k,idim-1))+
     +    (1.-bci(j,k,2))*vi0(j,k,1,3)
      enddo
      enddo
      end if
c If this is 1st global subiteration for time-accurate computation,
c save zksav (at time step n):
c zksav2(j,k,i,1) and (j,k,i,2) are turb quantities
c zksav2(j,k,i,3) and (j,k,i,4) are Delta turb quantities
      if (real(dt) .gt. 0. .and. icyc .eq. 1) then
      if (abs(ita) .eq. 2) then
c     if zksav2 at 1st point is zero, then we know that we do not have
c     2nd order data from the restart; no choice but to set
c     zksav2(j,k,i,3&4)=deltaQ=0 for 1st iteration
        if (real(zksav2(1,1,1,1)) .eq. 0.) then
        do i=1,idim-1
          do k=1,kdim-1
            do j=1,jdim-1
              zksav2(j,k,i,3)=0.
              zksav2(j,k,i,4)=0.
            enddo
          enddo
        enddo
        else
        do i=1,idim-1
          do k=1,kdim-1
            do j=1,jdim-1
              zksav2(j,k,i,3)=zksav(j,k,i,1)-zksav2(j,k,i,1)
              zksav2(j,k,i,4)=zksav(j,k,i,2)-zksav2(j,k,i,2)
            enddo
          enddo
        enddo
        end if
      end if
        do i=1,idim-1
          do k=1,kdim-1
            do j=1,jdim-1
              zksav2(j,k,i,1)=zksav(j,k,i,1)
              zksav2(j,k,i,2)=zksav(j,k,i,2)
            enddo
          enddo
        enddo
      end if
c Get TURRE values
      do i=1,idim-1
        do k=1,kdim-1
          do j=1,jdim-1
            turre(j,k,i,1)=zksav(j,k,i,1)
            turre(j,k,i,2)=zksav(j,k,i,2)
          enddo
        enddo
      enddo
c

c

c   for wall functions, force k & omega (or epsilon) in 1st cell off solid wall
c     wall function in k-direction
      if(iwf(3) .eq. 1) then
      do 9181 kk=1,2
      if(kk .eq. 1) then
        k=1
        kstop=nbck0(nbl)
      else
        k=kdim-1
        kstop=nbckdim(nbl)
      end if
      do 9171 kset=1,kstop
        if(abs(kbcinfo(nbl,kset,1,kk)).eq.2004 .or.
     .     abs(kbcinfo(nbl,kset,1,kk)).eq.2024 .or.
     .     abs(kbcinfo(nbl,kset,1,kk)).eq.2034 .or.
     .     abs(kbcinfo(nbl,kset,1,kk)).eq.2016) then
          ibeg=kbcinfo(nbl,kset,2,kk)
          iend=kbcinfo(nbl,kset,3,kk)-1
          jbeg=kbcinfo(nbl,kset,4,kk)
          jend=kbcinfo(nbl,kset,5,kk)-1
          do i=ibeg,iend
            do j=jbeg,jend
              if ((i.ge.ilamlo .and. i.lt.ilamhi .and.
     .             j.ge.jlamlo .and. j.lt.jlamhi .and.
     .             k.ge.klamlo .and. k.lt.klamhi) .or.
     .             real(smin(j,k,i)) .lt. 0.) then
                continue
              else
                uu  = sqrt((q(j,k,i,2)-qk0(j,i,2,kk+kk-1))**2 +
     +                     (q(j,k,i,3)-qk0(j,i,3,kk+kk-1))**2 +
     +                     (q(j,k,i,4)-qk0(j,i,4,kk+kk-1))**2 )
                dudy=uu/ccabs(smin(j,k,i))
                tauw=(fnu(j,k,i)+vk0(j,i,1,kk+kk-1))*dudy
                utau=sqrt(tauw/q(j,k,i,1)/re)
                omegatemp=utau/(sqrt(cmuc1)*vk*ccabs(smin(j,k,i))*re)
                omegatemp=ccmax(omegatemp,tur10(1))
                zktemp=utau*utau/sqrt(cmuc1)
                vist3d(j,k,i)=q(j,k,i,1)*zktemp/omegatemp
                if(ivmx .eq. 6 .or. ivmx .eq. 7) then
                  zksav(j,k,i,1)=omegatemp
                  zksav(j,k,i,2)=zktemp
                else if(ivmx .eq. 8 .or. ivmx .eq. 12 .or.
     +                  ivmx .eq. 14) then
                  zksav(j,k,i,1)=omegatemp*cmuc1
                  zksav(j,k,i,2)=zktemp
                else if(ivmx .eq. 15) then
                  zksav(j,k,i,1)=omegatemp*cmuc1*zktemp*q(j,k,i,1)/
     +              fnu(j,k,i)
                  zksav(j,k,i,2)=zktemp
                else
                  zksav(j,k,i,1)=omegatemp*cmuc1*zktemp
                  zksav(j,k,i,2)=zktemp
                end if
              end if
            enddo
          enddo
        end if
 9171 continue
 9181 continue
      end if
c     wall function in j-direction
      if(iwf(2) .eq. 1) then
      do 9182 jj=1,2
      if(jj .eq. 1) then
        j=1
        jstop=nbcj0(nbl)
      else
        j=jdim-1
        jstop=nbcjdim(nbl)
      end if
      do 9172 jset=1,jstop
        if(abs(jbcinfo(nbl,jset,1,jj)).eq.2004 .or.
     .     abs(jbcinfo(nbl,jset,1,jj)).eq.2024 .or.
     .     abs(jbcinfo(nbl,jset,1,jj)).eq.2034 .or.
     .     abs(jbcinfo(nbl,jset,1,jj)).eq.2016) then
          ibeg=jbcinfo(nbl,jset,2,jj)
          iend=jbcinfo(nbl,jset,3,jj)-1
          kbeg=jbcinfo(nbl,jset,4,jj)
          kend=jbcinfo(nbl,jset,5,jj)-1
          do i=ibeg,iend
            do k=kbeg,kend
              if ((i.ge.ilamlo .and. i.lt.ilamhi .and.
     .             j.ge.jlamlo .and. j.lt.jlamhi .and.
     .             k.ge.klamlo .and. k.lt.klamhi) .or.
     .             real(smin(j,k,i)) .lt. 0.) then
                continue
              else
                uu  = sqrt((q(j,k,i,2)-qj0(k,i,2,jj+jj-1))**2 +
     +                     (q(j,k,i,3)-qj0(k,i,3,jj+jj-1))**2 +
     +                     (q(j,k,i,4)-qj0(k,i,4,jj+jj-1))**2 )
                dudy=uu/ccabs(smin(j,k,i))
                tauw=(fnu(j,k,i)+vj0(k,i,1,jj+jj-1))*dudy
                utau=sqrt(tauw/q(j,k,i,1)/re)
                omegatemp=utau/(sqrt(cmuc1)*vk*ccabs(smin(j,k,i))*re)
                omegatemp=ccmax(omegatemp,tur10(1))
                zktemp=utau*utau/sqrt(cmuc1)
                vist3d(j,k,i)=q(j,k,i,1)*zktemp/omegatemp
                if(ivmx .eq. 6 .or. ivmx .eq. 7) then
                  zksav(j,k,i,1)=omegatemp
                  zksav(j,k,i,2)=zktemp
                else if(ivmx .eq. 8 .or. ivmx .eq. 12 .or.
     +                  ivmx .eq. 14) then
                  zksav(j,k,i,1)=omegatemp*cmuc1
                  zksav(j,k,i,2)=zktemp
                else if(ivmx .eq. 15) then
                  zksav(j,k,i,1)=omegatemp*cmuc1*zktemp*q(j,k,i,1)/
     +              fnu(j,k,i)
                  zksav(j,k,i,2)=zktemp
                else
                  zksav(j,k,i,1)=omegatemp*cmuc1*zktemp
                  zksav(j,k,i,2)=zktemp
                end if
              end if
            enddo
          enddo
        end if
 9172 continue
 9182 continue
      end if
c     wall function in i-direction
      if(i2d .ne. 1 .and. iwf(1) .eq. 1 .and. iaxi2planeturb.ne.1) then
      do 9183 ii=1,2
      if(ii .eq. 1) then
        i=1
        istop=nbci0(nbl)
      else
        i=idim-1
        istop=nbcidim(nbl)
      end if
      do 9173 iset=1,istop
        if(abs(ibcinfo(nbl,iset,1,ii)).eq.2004 .or.
     .     abs(ibcinfo(nbl,iset,1,ii)).eq.2024 .or.
     .     abs(ibcinfo(nbl,iset,1,ii)).eq.2034 .or.
     .     abs(ibcinfo(nbl,iset,1,ii)).eq.2016) then
          jbeg=ibcinfo(nbl,iset,2,ii)
          jend=ibcinfo(nbl,iset,3,ii)-1
          kbeg=ibcinfo(nbl,iset,4,ii)
          kend=ibcinfo(nbl,iset,5,ii)-1
          do j=jbeg,jend
            do k=kbeg,kend
              if ((i.ge.ilamlo .and. i.lt.ilamhi .and.
     .             j.ge.jlamlo .and. j.lt.jlamhi .and.
     .             k.ge.klamlo .and. k.lt.klamhi) .or.
     .             real(smin(j,k,i)) .lt. 0.) then
                continue
              else
                uu  = sqrt((q(j,k,i,2)-qi0(k,i,2,ii+ii-1))**2 +
     +                     (q(j,k,i,3)-qi0(k,i,3,ii+ii-1))**2 +
     +                     (q(j,k,i,4)-qi0(k,i,4,ii+ii-1))**2 )
                dudy=uu/ccabs(smin(j,k,i))
                tauw=(fnu(j,k,i)+vi0(k,i,1,ii+ii-1))*dudy
                utau=sqrt(tauw/q(j,k,i,1)/re)
                omegatemp=utau/(sqrt(cmuc1)*vk*ccabs(smin(j,k,i))*re)
                omegatemp=ccmax(omegatemp,tur10(1))
                zktemp=utau*utau/sqrt(cmuc1)
                vist3d(j,k,i)=q(j,k,i,1)*zktemp/omegatemp
                if(ivmx .eq. 6 .or. ivmx .eq. 7) then
                  zksav(j,k,i,1)=omegatemp
                  zksav(j,k,i,2)=zktemp
                else if(ivmx .eq. 8 .or. ivmx .eq. 12 .or.
     +                  ivmx .eq. 14) then
                  zksav(j,k,i,1)=omegatemp*cmuc1
                  zksav(j,k,i,2)=zktemp
                else if(ivmx .eq. 15) then
                  zksav(j,k,i,1)=omegatemp*cmuc1*zktemp*q(j,k,i,1)/
     +              fnu(j,k,i)
                  zksav(j,k,i,2)=zktemp
                else
                  zksav(j,k,i,1)=omegatemp*cmuc1*zktemp
                  zksav(j,k,i,2)=zktemp
                end if
              end if
            enddo
          enddo
        end if
 9173 continue
 9183 continue
      end if
c
      if (i_lam_forcezero .eq. 1) then
        do i=1,idim-1
          do k=1,kdim-1
            do j=1,jdim-1
              if ((i.ge.ilamlo .and. i.lt.ilamhi .and.
     .             j.ge.jlamlo .and. j.lt.jlamhi .and.
     .             k.ge.klamlo .and. k.lt.klamhi) .or.
     .             real(smin(j,k,i)) .lt. 0.) then
                vist3d(j,k,i)=0.
              end if
            enddo
          enddo
        enddo
      end if
c

c
      return
      end
