library(ggplot2) 
speed <- read.csv("output.csv", header=T)
speed$total_pixels = speed$xres * speed$yres
speed$real_seconds = speed$real_time/1000
speed$render_seconds = speed$render_time/1000
speed$megapixels = speed$total_pixels/1000000
speed$total_threads = speed$xthreads * speed$ythreads
levels(speed$total_threads) =  c(1,4,16,64,144,256,400,484) 

p <- ggplot(speed, aes(megapixels, legend = TRUE)) 
p <- p +  opts(title = "Performance of Raytracer Implementations at Various Resolutions", legend.title = theme_text(colour='black', size=10, face='bold', hjust=-.1))
p <- p + geom_jitter(aes(y = speed$real_time, shape = application_type, fill = factor(total_threads)), position = position_jitter(width=7)) + scale_shape("Run", solid=FALSE) + scale_color_hue("Application Type")
p <- p + scale_y_continuous("Render Time (ms)", formatter="comma", breaks = c(10000,60000,120000,180000,300000,420000)) 
p <- p + scale_x_continuous("Megapixels Rendered", formatter="comma", breaks = c(1,10,50,100,200,300,400,500)) 

ggsave("cumulative.pdf")

threads_1   <- subset(speed,total_threads==1)
threads_4   <- subset(speed,total_threads==4)
threads_16  <- subset(speed,total_threads==16)
threads_64  <- subset(speed,total_threads==64)
threads_144 <- subset(speed,total_threads==144)
threads_256 <- subset(speed,total_threads==256)
threads_400 <- subset(speed,total_threads==400)
threads_484 <- subset(speed,total_threads==484)

p <- ggplot(threads_1, aes(megapixels, legend = TRUE)) 
p <- p +  opts(title = "Run 1", legend.title = theme_text(colour='black', size=10, face='bold', hjust=-.1))
p <- p + geom_jitter(aes(y = threads_1$render_time, fill = application_type), position = position_jitter(width=10))  
p <- p + scale_y_continuous("Render Time (ms)", formatter="comma", breaks = c(10000,60000,120000,180000,300000,420000)) 
p <- p + scale_x_continuous("Megapixels Rendered", formatter="comma", breaks = c(1,10,50,100,200,300,400,500)) 
ggsave("threads_1.pdf")


p <- ggplot(speed, aes(megapixels, legend = TRUE)) 
p <- p +  opts(title = "Performance of Raytracer Implementations at Various Resolutions", legend.title = theme_text(colour='black', size=10, face='bold', hjust=-.1))
p <- p + geom_jitter(aes(y = speed$render_time, fill = factor(total_threads), shape = application_type), position = position_jitter(width=7)) + scale_shape("Run", solid=FALSE) + scale_color_hue("Application Type")
p <- p + scale_y_continuous("Render Time (ms)", formatter="comma", limits = c(0,10000)) 
p <- p + scale_x_continuous("Megapixels Rendered", formatter="comma", breaks = c(1,2,3,4,5,6,7,8,9,10), limits = c(0,10))
ggsave("inset.pdf")
