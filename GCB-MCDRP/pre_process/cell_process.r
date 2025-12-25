# 清除当前环境中的所有对象
rm(list = ls())
# 设置字符串处理方式，避免将字符串自动转换为因子
options(stringsAsFactors = F)

# 加载需要的R包
library(GSVA)       # GSVA包，用于基因集变异分析
library(GSEABase)   # GSEABase包，处理基因集（Gene Set）相关数据
library(msigdbr)    # msigdbr包，提供MSigDB（分子细胞信息基因集）数据
library(clusterProfiler)  # clusterProfiler包，用于富集分析
library(org.Hs.eg.db)     # org.Hs.eg.db包，提供人类基因注释数据
library(enrichplot)   # enrichplot包，生成富集分析的可视化结果
library(limma)       # limma包，进行差异分析和其他相关的统计分析

# 删除环境中的所有对象，确保没有旧的数据或对象干扰
remove(list = ls())

# 读取基因表达数据
mydata <- read.table("../gene_expressions.csv", sep=",",  # 读取csv文件
                    header=T, row.names=1, check.names=F)  # 数据框的第一行为列名，第一列为行名
# 转置数据框：原数据的行和列互换，转置后每一行为细胞系，每一列为基因
mydata <- t(mydata)
# 将数据框转化为矩阵格式，方便后续计算
mydata = as.matrix(mydata)

# 打印前5行5列数据，以检查读取的数据
mydata[1:5, 1:5]

# 导入MSigDB（Molecular Signatures Database）数据集的路径
msigdb <- "../c2.cp.v2022.1.Hs.symbols.gmt"

# 读取MSigDB的GMT格式文件，这个文件包含了很多基因集，每个基因集对应一组基因
geneset <- getGmt(file.path(msigdb))

# 执行GSVA分析
# gsva()函数会对每个样本计算其在基因集上的评分（GSVA评分）
# 其中mydata是输入的基因表达矩阵，geneset是包含基因集的对象
# 参数说明：
# min.sz=5：基因集最小大小为5个基因，少于5个基因的基因集会被忽略
# mx.diff=FALSE：默认选择的是“最大差异”模式，即选择最具差异性的基因集进行评分
# kcdf="Gaussian"：指定数据分布的类型（这里是高斯分布）
# parallel.sz：并行计算使用的核心数目，parallel::detectCores()返回可用的核心数
es.max <- gsva(mydata, geneset, min.sz=5,
               mx.diff=FALSE, verbose=T, kcdf="Gaussian",
               parallel.sz = parallel::detectCores())

# 打印GSVA分析结果的前10行和前2列（即每个基因集的GSVA评分）
head(es.max[1:10,1:2])

# 将GSVA分析结果保存为一个文本文件（以制表符分隔）
write.table(es.max, file="../GSVA_deepcdr_result.txt", sep="\t",  # 保存为文本文件
            quote=F, row.names = T)  # 不加引号，行名保留

# 保存GSVA结果为R的数据格式文件（.Rda），方便后续的读取
save(es.max, file = "../GSVA_cosmic_result.Rda")


