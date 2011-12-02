library(ggplot2)
gprof_output <- read.csv("gprof_master.csv",header=TRUE)
gprof_output$resolution <- factor(gprof_output$xres * gprof_output$yres, 
                                  labels = c("800x800", "1600x1600", "2400x2400", "3200x3200", "4000x4000", "4800x4800")
                                  ) 
gprof_output$percent_time <- gprof_output$percent_time/100

p <- ggplot(gprof_output, aes(x = name, y = percent_time, fill = name)) + 
#opts(axis.title.x = theme_blank(), axis.text.x = theme_blank(), strip.title.x = theme_blank(), strip.title.y = theme_blank()) + 
opts(axis.text.x = theme_blank()) + 
geom_bar(width = 1) + scale_x_discrete("Output Resolution (pixels)") + scale_fill_hue("Function") + scale_y_continuous("Percentage of Running Time", formatter = "percent") + facet_grid(. ~ resolution)
p
