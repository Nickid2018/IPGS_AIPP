package io.github.nickid2018.ipgs;

import io.github.nickid2018.ipgs.network.U2Net;
import org.junit.jupiter.api.Test;

public class U2NetTest {

    @Test
    public void test() {
        System.setProperty("org.bytedeco.javacpp.maxPhysicalBytes", "3G");
        U2Net.init(2048, 2048, 3);
    }
}
