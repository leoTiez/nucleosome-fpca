% Cluster all genes then filter by gene size
load ('/data_directory/gene_size.mat');
ORF_g = ORF(gene_size>=1000);
name_dir=dir('/data_directory/*.bam');
for i =1:9
    data_dir = [char(name_dir(i).folder),'/',char(name_dir(i).name)];
    data_work = [char(name_dir(i).folder),'/cluster_all_ge_1000/',char(name_dir(i).name)];
    mkdir(data_work);
    load([data_dir,'/Classification.mat']);
    load([data_dir,'/AlignedProfile.mat']);
    [C1,ia,ib]=intersect(ORF_g,ORF);
    to_save =[data_work,'/','AlignedProfile.txt'];
    AlignedProfile_g = AlignedProfile(ib,:);
    save(to_save,'AlignedProfile_g','-ascii');
    %
    E = evalclusters(AlignedProfile,'kmeans','silhouette','klist',1:10);
    Fig1=figure(1);plot(E);
    %
    rng('default');
    
    [cidx, ctrs] = kmeans(AlignedProfile,E.OptimalK,'dist','corr','rep',20,'disp','final');
    cidx_g = cidx(ib);
    Fig2=figure(2);
    for c = 1:E.OptimalK
        subplot(E.OptimalK,1,c);
        plot(AlignedProfile_g((cidx_g == c),:)');
        axis tight
        xlabel('x (bp)')
        
    end
    suptitle('K-Means Clustering of Profiles');
    %
    Fig3=figure(3);
    for c = 1:E.OptimalK
        subplot(E.OptimalK,1,c);
        plot(ctrs(c,:)');
        axis tight
        axis off
    end
    suptitle('K-Means Clustering of Profiles');
    %
    to_save =[data_work,'/','cidx.txt'];
    save(to_save,'cidx_g','-ascii');
    %
    to_save =[data_work,'/','cluster_mean_profile.txt'];
    save(to_save,'ctrs','-ascii');
    %
    to_save =[data_work,'/','ORF.txt'];
    fid = fopen(to_save ,'w');
    fprintf(fid,'%s\n', C1{:});
    fclose(fid);
    %
    to_save = [data_work,'/silouhette.png'];
    saveas(Fig1,to_save);
    to_save = [data_work,'/silouhette.fig'];
    saveas(Fig1,to_save);
    to_save = [data_work,'/cluster.png'];
    saveas(Fig2,to_save);
    to_save = [data_work,'/cluster.fig'];
    saveas(Fig2,to_save);
    to_save = [data_work,'/centroid.png'];
    saveas(Fig3,to_save);
    to_save = [data_work,'/centroid.fig'];
    saveas(Fig3,to_save);
end